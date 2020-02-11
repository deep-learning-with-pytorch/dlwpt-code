package de.lernapparat.zebraify;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends AppCompatActivity {
    static final int REQUEST_IMAGE_CAPTURE = 1;
    private org.pytorch.Module model;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_main);

        TextView tv= (TextView) findViewById(R.id.headline);
        tv.setOnClickListener(new View.OnClickListener() {
            public void onClick(View v) {
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                // takePictureIntent.putExtra(android.provider.MediaStore.EXTRA_OUTPUT, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                if (takePictureIntent.resolveActivity(getPackageManager()) != null) {
                    startActivityForResult(takePictureIntent, REQUEST_IMAGE_CAPTURE);
                }
            }
        });


        try {
            model = Module.load(assetFilePath(this, "traced_zebra_model.pt"));
        } catch (IOException e) {
            Log.e("Zebraify", "Error reading assets", e);
            finish();
        }

    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
            // this gets called when the camera app got a picture
            Bitmap bitmap = (Bitmap) data.getExtras().get("data");

            final float[] means = {0.0f, 0.0f, 0.0f};
            final float[] stds = {1.0f, 1.0f, 1.0f};
            // preparing input tensor
            final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(bitmap,
                    means, stds);

            // running the model
            final Tensor outputTensor = model.forward(IValue.from(inputTensor)).toTensor();
            Bitmap output_bitmap = tensorToBitmap(outputTensor, means, stds, Bitmap.Config.RGB_565);

            ImageView image_view = (ImageView) findViewById(R.id.imageView);
            image_view.setImageBitmap(output_bitmap);
        }
    }

    // This is intended to be the inverse of bitmapToFloat32Tensor
    static Bitmap tensorToBitmap(Tensor tensor, float[] normMeanRGB, float[] normStdRGB, Bitmap.Config bc) {
        final float[] outputArray = tensor.getDataAsFloatArray();
        final long[] shape = tensor.shape();
        int width = (int) shape[shape.length - 1];
        int height = (int) shape[shape.length - 2];
        Bitmap output_bitmap = Bitmap.createBitmap(width, height, bc);

        int numPixels = width * height;
        int[] pixels = new int[numPixels];
        for (int i = 0; i < numPixels; i++) {
            pixels[i] = ((int) ((outputArray[0 * numPixels + i] * normStdRGB[0] + normMeanRGB[0]) * 255 + 0.49999) << 16)
                      + ((int) ((outputArray[1 * numPixels + i] * normStdRGB[1] + normMeanRGB[1]) * 255 + 0.49999) << 8)
                      + ((int) ((outputArray[2 * numPixels + i] * normStdRGB[2] + normMeanRGB[2]) * 255 + 0.49999));
        }
        output_bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return output_bitmap;
    }

    /**
     * Taken from PyTorch's HelloWorld Android app.
     *
     * Copies specified asset to the file in /files app directory and returns this file absolute path.
     *
     * @return absolute file path
     */
    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (false && file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file, false)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}
