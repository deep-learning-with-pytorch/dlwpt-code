import sys
import asyncio
import itertools
import functools
from sanic import Sanic
from sanic.response import  json, text
from sanic.log import logger
from sanic.exceptions import ServerError

import sanic
import threading
import PIL.Image
import io
import torch
import torchvision
from .cyclegan import get_pretrained_model

app = Sanic(__name__)

device = torch.device('cpu')
# we only run 1 inference run at any time (one could schedule between several runners if desired)
MAX_QUEUE_SIZE = 3  # we accept a backlog of MAX_QUEUE_SIZE before handing out "Too busy" errors
MAX_BATCH_SIZE = 2  # we put at most MAX_BATCH_SIZE things in a single batch
MAX_WAIT = 1        # we wait at most MAX_WAIT seconds before running for more inputs to arrive in batching

class HandlingError(Exception):
    def __init__(self, msg, code=500):
        super().__init__()
        self.handling_code = code
        self.handling_msg = msg

class ModelRunner:
    def __init__(self, model_name):
        self.model_name = model_name
        self.queue = []

        self.queue_lock = None

        self.model = get_pretrained_model(self.model_name,
                                          map_location=device)

        self.needs_processing = None

        self.needs_processing_timer = None

    def schedule_processing_if_needed(self):
        if len(self.queue) >= MAX_BATCH_SIZE:
            logger.debug("next batch ready when processing a batch")
            self.needs_processing.set()
        elif self.queue:
            logger.debug("queue nonempty when processing a batch, setting next timer")
            self.needs_processing_timer = app.loop.call_at(self.queue[0]["time"] + MAX_WAIT, self.needs_processing.set)

    async def process_input(self, input):
        our_task = {"done_event": asyncio.Event(loop=app.loop),
                    "input": input,
                    "time": app.loop.time()}
        async with self.queue_lock:
            if len(self.queue) >= MAX_QUEUE_SIZE:
                raise HandlingError("I'm too busy", code=503)
            self.queue.append(our_task)
            logger.debug("enqueued task. new queue size {}".format(len(self.queue)))
            self.schedule_processing_if_needed()

        await our_task["done_event"].wait()
        return our_task["output"]

    def run_model(self, batch):  # runs in other thread
        return self.model(batch.to(device)).to('cpu')

    async def model_runner(self):
        self.queue_lock = asyncio.Lock(loop=app.loop)
        self.needs_processing = asyncio.Event(loop=app.loop)
        logger.info("started model runner for {}".format(self.model_name))
        while True:
            await self.needs_processing.wait()
            self.needs_processing.clear()
            if self.needs_processing_timer is not None:
                self.needs_processing_timer.cancel()
                self.needs_processing_timer = None
            async with self.queue_lock:
                if self.queue:
                    longest_wait = app.loop.time() - self.queue[0]["time"]
                else:  # oops
                    longest_wait = None
                logger.debug("launching processing. queue size: {}. longest wait: {}".format(len(self.queue), longest_wait))
                to_process = self.queue[:MAX_BATCH_SIZE]
                del self.queue[:len(to_process)]
                self.schedule_processing_if_needed()
            # so here we copy, it would be neater to avoid this
            batch = torch.stack([t["input"] for t in to_process], dim=0)
            # we could delete inputs here...

            result = await app.loop.run_in_executor(
                None, functools.partial(self.run_model, batch)
            )
            for t, r in zip(to_process, result):
                t["output"] = r
                t["done_event"].set()
            del to_process

style_transfer_runner = ModelRunner(sys.argv[1])

@app.route('/image', methods=['PUT'], stream=True)
async def image(request):
    try:
        print (request.headers)
        content_length = int(request.headers.get('content-length', '0'))
        MAX_SIZE = 2**22 # 10MB
        if content_length:
            if content_length > MAX_SIZE:
                raise HandlingError("Too large")
            data = bytearray(content_length)
        else:
            data = bytearray(MAX_SIZE)
        pos = 0
        while True:
            # so this still copies too much stuff.
            data_part = await request.stream.read()
            if data_part is None:
                break
            data[pos: len(data_part) + pos] = data_part
            pos += len(data_part)
            if pos > MAX_SIZE:
                raise HandlingError("Too large")

        # ideally, we would minimize preprocessing...
        im = PIL.Image.open(io.BytesIO(data))
        im = torchvision.transforms.functional.resize(im, (228, 228))
        im = torchvision.transforms.functional.to_tensor(im)
        im = im[:3]  # drop alpha channel if present
        if im.dim() != 3 or im.size(0) < 3 or im.size(0) > 4:
            raise HandlingError("need rgb image")
        out_im = await style_transfer_runner.process_input(im)
        out_im = torchvision.transforms.functional.to_pil_image(out_im)
        imgByteArr = io.BytesIO()
        out_im.save(imgByteArr, format='JPEG')
        return sanic.response.raw(imgByteArr.getvalue(), status=200,
                                  content_type='image/jpeg')
    except HandlingError as e:
        # we don't want these to be logged...
        return sanic.response.text(e.handling_msg, status=e.handling_code)

app.add_task(style_transfer_runner.model_runner())
app.run(host="0.0.0.0", port=8000,debug=True)

