import os
import sys
import re
import datetime
import pathlib
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from Z01_1_General import removeSysFiles, DATEFORMAT
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def fallingRateDetection(df):
    fallingrate = 0
    THRESHOLD = 5.5
    humid_columns = [x for x in df.columns if 'humid' in x]
    df = df[humid_columns]
    df = df[df > 0]
    dflen = len(df)
    baseline = df.index[0] + datetime.timedelta(hours=6)
    baseline_average = df.loc[:baseline, :].mean().mean()
    window_start = baseline
    window_end = baseline + datetime.timedelta(hours=1)
    count = 0
    while window_start < df.index[-1]:
        baseline_average_now = df.loc[df.index[0] + datetime.timedelta(minutes=count):window_start, :].mean().mean()
        if baseline_average_now > baseline_average:
            baseline_average = baseline_average_now

        current_average = df.loc[window_start:window_end, :].mean().mean()

        if (baseline_average - current_average) > THRESHOLD:
            fallingrate = window_start
            break

        window_start = window_start + datetime.timedelta(minutes=30)
        window_end = window_start + datetime.timedelta(hours=1)
        count += 30
    return fallingrate

def parsedate(df, formats, check):
    for format in formats:
        status = 0
        try:
            dfdate = datetime.datetime.strptime(df['timestamp'][0], format)
            if check.split('-')[1] == '2022':
                datecheck = datetime.datetime.strptime(
                    '-'.join(check.split('\\')[-1].split('.')[0].split(' ')[0].split('-')[1:]), '%Y-%m-%d')
            else:
                datecheck = datetime.datetime.strptime(check.split('-')[1], '%y%m%d')
            if dfdate.date() == datecheck.date() or dfdate.date() == (
                    datecheck + datetime.timedelta(days=1)).date():
                df['timestamp'] = pd.to_datetime(df['timestamp'], format=format)
                return df, status
        except ValueError:
            pass
            status = 1
    return df, status
def loadDataset(filepath, formats):
    status = 0
    df = pd.read_csv(filepath)
    dcheck = filepath.split('/')[-1].split('.')[0]
    df, status = parsedate(df, formats, dcheck)
    df = df.dropna(subset=['timestamp'])
    if status != 1:
        df = df.set_index('timestamp')
        df = df.sort_index()
        df.index = df.index.map(lambda x: x.replace(second=0))
        df = df.dropna(how='all')
        df = df[~df.index.duplicated(keep='first')]
    else:
        print('failed')
    return df, status

def prepareData(rawDataDir, preparedDataDir, boMsMdl=False, lookBckwd=datetime.timedelta(minutes=0), boGenFullData=False):

    # Delete existing prepared data before creating latest
    if os.path.isdir(preparedDataDir):
        shutil.rmtree(preparedDataDir)

    mc_files = os.listdir(f"{rawDataDir}/mc data")
    mc_files = removeSysFiles(mc_files)

    total = len(mc_files)
    counter = 0
    status = 0

    for file in mc_files:
            counter = counter + 1
            progress = (counter / total) * 100
            print(f"Data processing progress: {progress:.2f}%", end="\r")
            sensor_df, status = loadDataset(f"{rawDataDir}/sensor data/{file}", DATEFORMAT)

            if status == 1:
                print(f"Warning[0]: {file} The date time is not parsed correctly, check the date format of the file.")
                return status

            # reindex so that every minute has a value
            # sensor data is extended to the same length as mc_df
            timestamp = pd.date_range(start=sensor_df.index[0], end=sensor_df.index[-1], freq='min')
            sensor_df = sensor_df.reindex(timestamp)


            # rearrange
            humid_columns = [x for x in sensor_df.columns if "humidity" in x]
            humid_columns = sorted(humid_columns, key=lambda x: int(x.split(' ')[1]))
            temp_columns = [x for x in sensor_df.columns if "temperature" in x]
            temp_columns = sorted(temp_columns, key=lambda x: int(x.split(' ')[1]))

            Sin_columns = [humid_columns[-2], temp_columns[-2]]
            Sout_columns = [humid_columns[-1], temp_columns[-1]]

            # group same data type together
            humid_df = sensor_df[humid_columns]
            temp_df = sensor_df[temp_columns]

            Sin_df = sensor_df[Sin_columns]
            Sout_df = sensor_df[Sout_columns]

            Sin_df = Sin_df.interpolate(method="time")
            Sout_df = Sout_df.interpolate(method="time")

            # transform to region average
            def sensorToRegion(df, dataname):
                num_cols = len(df.columns)
                NUM_SENSOR_PER_REGION = 3
                num_region = int(num_cols / NUM_SENSOR_PER_REGION)

                # calculate each region's average
                start = 0
                end = 0
                result_df = pd.DataFrame(index=df.index)

                for i in range(num_region):
                    end = start + NUM_SENSOR_PER_REGION
                    col_name = f"{dataname}{str(i)}"

                    average = df.iloc[:, start:end].mean(axis=1, skipna=True).interpolate(method='time')
                    result_df[col_name] = average.to_numpy().flatten()

                    start = start + NUM_SENSOR_PER_REGION

                return result_df

            humid_df = sensorToRegion(humid_df, "humid")
            temp_df = sensorToRegion(temp_df, "temp")

            # PATCH 20231115: backfill nan in first row
            humid_df = humid_df.bfill()
            temp_df = temp_df.bfill()

            # combine humidity and temperature df
            sensor_df = pd.concat([humid_df, temp_df, Sin_df, Sout_df], axis=1)

            sensor_df = sensor_df.bfill()

            # smoothen sensor df
            def mySavgolFilter(x):
                return savgol_filter(x, 121, 2)
            sensor_df = sensor_df.apply(mySavgolFilter)

            df = sensor_df

            # rename index name
            df.index.name = 'timestamp'
            def finalizeNSaveData(df, p2DataDir):
                # separate back to mc and sensor
                sensor_columns = [x for x in df.columns if "mc" not in x]
                sensor_df = df[sensor_columns]

                # plot graph to check
                def plotGraph(filepath, filename, sensor_df):
                    # print(f'Plot graph filepath: {filepath}')
                    # plot the data to view
                    fig, ax = plt.subplots(figsize=(20, 15))
                    # to plot without minute, remove minute from column list
                    columns = [x for x in sensor_df.columns if 'minute' not in x]
                    ax.plot(sensor_df[columns], color='blue')
                    ax2 = ax.twinx()
                    ax.set_ylim((0, 100))
                    ax2.set_ylim((10, 28))
                    plt.title(filename)
                    plt.grid()
                    # plt.show()
                    plt.savefig(filepath, facecolor='white', transparent=False)
                    plt.close()

                plot_output_path = f"{p2DataDir}/plots"
                pathlib.Path(plot_output_path).mkdir(parents=True, exist_ok=True)
                graph_path = f"{plot_output_path}/{file[:-4]}"
                plotGraph(graph_path, file[:-4], sensor_df)
                # print(f"Processed data plotted under {graph_path}")

                # export as csv
                pathlib.Path(f"{p2DataDir}/mc data").mkdir(parents=True, exist_ok=True)
                pathlib.Path(f"{p2DataDir}/sensor data").mkdir(parents=True, exist_ok=True)
                sensor_df.to_csv(f"{p2DataDir}/sensor data/{file}")

            # Save dfs
            p2DataDir = f"{preparedDataDir}/{'msMdlData' if boMsMdl else ''}"
            finalizeNSaveData(df, p2DataDir)


        # except Exception as e:
        #     print(file + "Fail")
        #     # print(mc_df)
        #     exc_type, exc_obj, exc_tb = sys.exc_info()
        #     fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        #     print(exc_type, fname, exc_tb.tb_lineno)

    if status == 1:
        cont = str(input("Continue training? (Y/N): "))
        if cont != "Y":
            quit()

    return status

def scaleData(df, scaler):
    if scaler != None:
        scaled_df = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled_df, columns = df.columns)
        return scaled_df
    else:
        scaler = MinMaxScaler()
        scaled_df = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_df, columns = df.columns)
        return scaled_df, scaler

def transformData(df, n_steps_in):
    start = 0
    x_end = 0
    x_data, y_data = [], []
    for n in range(len(df)):
        x_end = start + n_steps_in
        y_end = x_end + n_steps_in
        if y_end >= len(df):
            break
        x = df.iloc[start:x_end, :].values
        y = df.iloc[x_end:y_end, :].values
        x_data.append(x)
        y_data.append(y)
        start = start+1
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

def transformDataSingleStep(df, n_steps_in):
    start = 0
    x_end = 0
    x_data, y_data = [], []
    for n in range(len(df)):
        x_end = start + n_steps_in
        if x_end >= len(df):
            break
        x = df.iloc[start:x_end, :].values
        y = df.iloc[x_end, :].values
        x_data.append(x)
        y_data.append(y)
        start = start+1
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    return x_data, y_data

def generateDatasets(testValidDataDir, processedDataDir):
    # Split Datasets
    train_data_files, test_data_files, valid_data_files = splitDatasets(testValidDataDir, processedDataDir)

    # Prepare Train / Test Datasets
    dfXTrain, dfYTrain = prepareDataset(processedDataDir, train_data_files)
    dfXTest, dfYTest = prepareDataset(processedDataDir, test_data_files)
    dfXValid, dfYValid = prepareDataset(processedDataDir, valid_data_files, boShowDt=True)

    print("Dataset description")
    print("x_train, y_train", dfXTrain.shape, dfYTrain.shape)
    print("x_test, y_test", dfXTest.shape, dfYTest.shape)
    print("x_valid, y_valid", dfXValid.shape, dfYValid.shape)

    return (dfXTrain, dfYTrain, dfXTest, dfYTest, dfXValid, dfYValid), (train_data_files, test_data_files, valid_data_files)

def splitDatasets(testValidDataDir, processedDataDir):
    testDataFile = f'{testValidDataDir}/test_data.txt'
    validDataFile = f'{testValidDataDir}/validation_data.txt'

    if os.path.exists(testDataFile):
        with open(testDataFile, "r") as file:
            test_data_files = file.read().splitlines()
    else:
        print("Text file containing test data file name is missing, please check before proceed.")
        quit()
    if os.path.exists(validDataFile):
        with open(validDataFile, "r") as file:
            valid_data_files = file.read().splitlines()
    else:
        print("Text file containing validation data file name is missing, please check before proceed.")
        quit()

    train_data_files = os.listdir(f"{processedDataDir}/sensor data")
    train_data_files = list(set(train_data_files)-set(test_data_files))
    train_data_files = list(set(train_data_files)-set(valid_data_files))

    return train_data_files, test_data_files, valid_data_files


def prepareDataset(rawDataDir, files, boShowFn=False, boShowDt=False):

    sensor_path = f'{rawDataDir}/sensor data'
    mc_path = f'{rawDataDir}/mc data'

    if '.ipynb_checkpoints' in files:
        files.remove('.ipynb_checkpoints')

    x = pd.read_csv(f"{sensor_path}/{files[0]}")
    y = pd.read_csv(f"{mc_path}/{files[0]}")

    if boShowFn:
        x.insert(0, 'fn', files[0])
        y.insert(0, 'fn', files[0])

    files = files[1:]

    for file in files:
        xi = pd.read_csv(f"{sensor_path}/{file}")
        yi = pd.read_csv(f"{mc_path}/{file}")

        if boShowFn:
            xi.insert(0, 'fn', file)
            yi.insert(0, 'fn', file)

        x = pd.concat([x, xi], axis=0)
        y = pd.concat([y, yi], axis=0)

        x['timestamp'] = pd.to_datetime(x['timestamp'])
        y['timestamp'] = pd.to_datetime(y['timestamp'])
        x.index = x['timestamp']
        y.index = y['timestamp']

    if not boShowDt:
        x = x.drop('timestamp', axis=1)
        y = y.drop('timestamp', axis=1)

    return x, y


def prepareMc(mc_path, files, boShowDt=False):
    # Demo of use #
    # yValidLstm = prepareMc(lstmResultsDir, valid_data_files, boShowDt=True)

    y = pd.read_csv(f"{mc_path}/{files[0]}")
    y.insert(0, 'fn', files[0])

    files = files[1:]

    for file in files:
        if file == '.ipynb_checkpoints':
            continue
        yi = pd.read_csv(f"{mc_path}/{file}")
        yi.insert(0, 'fn', file)
        y = pd.concat([y, yi], axis=0)

    y['timestamp'] = pd.to_datetime(y['timestamp'])
    y.index = y['timestamp']
    if boShowDt:
        y = y.drop('timestamp', axis=1)

    return y


def getRecsForFn(df, fn):
    # Demo of Use #
    # xValid, yValid = getRecsForFn(dfXValid_Ms, dfYValid_Ms, fn)

    x = df.loc[df['fn'] == fn, :]

    x = x.drop("fn", axis=1)

    for dtCol in [c for c in x.columns if re.match('s\d+_timestamp', c) or re.match('timestamp', c)]:
        x[dtCol] = pd.to_datetime(x[dtCol])
    if 'timestamp' in x.columns:
        x.index = x['timestamp']

    return x