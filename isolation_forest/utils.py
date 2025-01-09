import os
import re
import pandas as pd
import numpy as np


def extract_numbers(filename):
    """파일 이름에서 실험 번호와 샘플 번호를 추출합니다.

    Args:
        filename (str): 파일 이름 (예: 'T1_1_Expt_1_2.csv')

    Returns:
        tuple: 추출된 숫자들 (T 번호, 샘플 번호, 실험 번호 1, 실험 번호 2)을 정수 형태로 반환합니다.
        None: 파일 이름이 형식에 맞지 않으면 None을 반환합니다.
    """
    """Extract numbers from filename: 'T#_#_Expt_#_#.csv'"""
    match = re.match(r'T(\d+)_(\d+)_Expt_(\d+)_(\d+).csv', filename)
    if match:
        T, S, E, _ = map(int, match.groups())
        return T, S, E
    print('No match')
    return None


def combine_data(dir_path, Tnum, sensor='Accelerometer'):
    """주어진 디렉토리에서 특정 센서 데이터를 통합하여 DataFrame으로 반환합니다.

    Args:
        dir_path (str): 데이터 파일이 저장된 디렉토리 경로
        Tnum (str): 실험 번호 디렉토리 (예: 'T1')
        sensor (str): 센서 유형 (예: 'Accelerometer', 기본값: 'Accelerometer')

    Returns:
        pandas.DataFrame: 통합된 센서 데이터
    """
    dfs = []
    path = os.path.join(dir_path, Tnum, sensor)
    if os.path.exists(path):
        sorted_files = sorted(os.listdir(
            path), key=lambda x: extract_numbers(x))
        for file in sorted_files:
            if file.endswith('.csv'):
                df = pd.read_csv(os.path.join(path, file))
                _, S, E = extract_numbers(file)
                df['sample'] = S
                df['Expt'] = E

                dfs.append(df)
    else:
        print(f'{path} not found')
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def downsampling(ACC, FOR):
    """두 데이터를 동일한 크기와 형태로 맞춰 비교하거나 입력으로 사용할 수 있도록 데이터를 변환

    Args:
        ACC (pandas.DataFrame): 가속도계 데이터
        FOR (pandas.DataFrame): 힘 센서 데이터

    Returns:
        pandas.DataFrame: 다운샘플링된 가속도계 데이터
    """
    dfs = []
    sample_max = int(FOR.iloc[-1]['sample'])  # Maximum sample value in FOR

    for s in range(1, sample_max + 1):
        # Filter data for the current sample
        sample_acc = ACC[ACC['sample'] == s]
        sample_for = FOR[FOR['sample'] == s]

        # Determine lengths for interpolation
        len_acc = len(sample_acc)
        len_for = len(sample_for)
        x = np.linspace(0, 1, len_acc)
        x_new = np.linspace(0, 1, len_for)

        # Drop unnecessary columns and interpolate
        sample_acc = sample_acc.drop(columns=['sample', 'Expt'])
        df = pd.DataFrame(
            {col: np.interp(x_new, x, sample_acc[col]) for col in sample_acc.columns})

        # Append interpolated DataFrame to the list
        dfs.append(df)

    # Concatenate all interpolated DataFrames
    ACC_interp = pd.concat(dfs, ignore_index=True)
    return ACC_interp


def preprocess_data(tnum, dir_path):
    Tx_acc_train = combine_data(dir_path, Tnum=tnum, sensor='Accelerometer')
    Tx_for_train = combine_data(dir_path, Tnum=tnum, sensor='Force')

    Tx_acc_interp_train = downsampling(Tx_acc_train, Tx_for_train)
    Tx_for_train = Tx_for_train.drop(columns=['sample'])

    Tx = pd.concat([Tx_acc_interp_train, Tx_for_train], axis=1)

    return Tx
