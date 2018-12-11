import copy
import csv

# instant,dteday,season,yr,mnth,hr,holiday,weekday,workingday,weathersit,temp,atemp,hum,windspeed,casual,registered,cnt
with open('hour.csv', newline='') as hour_file, open('hour-fixed.csv', 'w', newline='') as fixed_file:
    hour_csv = csv.reader(hour_file)
    fixed_csv = csv.writer(fixed_file)

    last_row = None
    for this_row in hour_csv:
        if last_row is None:
            pass
        elif last_row[0] == 'instant':
            pass
        else:
            last_hour = int(last_row[5])
            this_hour = int(this_row[5])

            if this_hour < last_hour:
                this_hour += 24

            missing_row = copy.deepcopy(last_row)
            missing_row[-1] = 0

            for missing_hour in range(last_hour+1, this_hour):
                if missing_hour == 24:
                    missing_row = copy.deepcopy(this_row)
                    missing_row[-1] = 0

                missing_row[5] = missing_hour % 24

                fixed_csv.writerow(missing_row)

                print(last_hour, this_hour, missing_row)

        fixed_csv.writerow(this_row)
        last_row = this_row
