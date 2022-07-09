# Create your views here.
from asyncore import read
from distutils.command.upload import upload
import re
from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
import os
import csv
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import time

import FinanceDataReader as fdr
from pykrx import stock
import pandas_datareader as wb

#def main(request):
#   return HttpResponse("하이.")
    
def main(request):
    return render(request, 'main.html')

result_cols = ['one_day_trade_result', 'seven_trade_result', 'fourteen_trade_result']

def ajax_csv(request):
    data_dict = {}

    if request.method == 'POST':
        #javascrpit에서 데이터를 가져와 DataFrame으로 변형(데이터분석을 위해)
        uploaded = request.POST.get('upload_data', None)
        uploaded_list = json.loads(uploaded)

        df = pd.DataFrame()
        for t in uploaded_list:
            tmp = pd.DataFrame(t, index=[0])
            df = pd.concat([df, tmp], axis=0)
        
        #전처리 - column rename, Nan 제거
        df = df.rename(columns={'date\r' : 'date'})
        df = df.dropna()
        
        df_anal = stock_holding_income(df)

        sell_all, half_sell, sell_div, final_sell = get_analyze(df_anal)
        
        data_dict['sell_all'] = sell_all
        data_dict['half_sell'] = half_sell
        data_dict['sell_div'] = sell_div
        data_dict['final_sell'] = final_sell
        data_dict['length'] = len(df_anal)
        data_dict['one_day_max'] = max(df_anal['one_day'])
        data_dict['one_day_min'] = min(df_anal['one_day'])
        data_dict['seven_days_max'] = max(df_anal['seven_days'])
        data_dict['seven_days_min'] = min(df_anal['seven_days'])
        data_dict['fourteen_days_max'] = max(df_anal['fourteen_days'])
        data_dict['fourteen_days_min'] = min(df_anal['fourteen_days'])
        data_dict['win_percentage'] = get_income_static(df_anal)

        # return render(request, 'main.html', data_dict)
        json_dict = json.dumps(data_dict) 
        # print(type(json_dict))
        return HttpResponse(json_dict)
        

def stock_holding_income(df):
    df.symbol = df.symbol.apply(lambda x : str(x).replace('A',''))
    df['date'] = df['date'].apply(lambda x : str(x).replace('\r', ''))
    
    seven_days_list = []
    day_after = []
    fourteen_list = []
    index_list = []
    
    for i in range((len(df['symbol']))):
        current_time = datetime.strptime(df['date'].iloc[i], '%Y-%m-%d')

        #다음날 수익률 체크
        target_time = current_time + timedelta(days = 14)
        target_time = str(target_time)[:10].replace('-', '')
        
        #인덱스 지수 확인 
        index_start_time = current_time + timedelta(days = -90)
        start_time = df['date'].iloc[i].replace('-','')

        current = 1
        one_day = 1
        seven_days = 1
        fourteen_days = 1
        
        one_day_income = 0
        seven_days_income = 0
        fourteen_days_income = 0
        
        index_status = -1
        
        try:
            partial_df = fdr.DataReader(df.symbol.iloc[i], start_time, target_time)
            
            #상승률 체크
            current = partial_df.Close[0]
            one_day = partial_df.High[1]
            seven_days = partial_df.High[-7]
            fourteen_days = partial_df.High[-1]
            
            one_day_income = ((float(one_day) - float(current)) / float(current) * 100)
            seven_days_income = ((seven_days - current) / float(current) * 100)
            fourteen_days_income = ((fourteen_days - current) / float(current) * 100)
            
            #인덱스 추출
            start = datetime(index_start_time.year, index_start_time.month, index_start_time.day)
            end = datetime(current_time.year, current_time.month, current_time.day)
            
            #야후를 통해서 코스피 지수 불러옴
            df_index = wb.DataReader("^KS11","yahoo", start, end)
            df_index = df_index.dropna()
            time.sleep(0.2)
            
            sixty = df_index.Close.rolling(window=60).mean()[-1]
            twenty = df_index.Close.rolling(window=20).mean()[-1]
            
            if sixty < twenty:
                index_status = 1
            else:
                index_status = 0
            
        except:
            print('error occured')
            pass
        
        #1일 7일 보유 수익률 확인
        day_after.append(one_day_income)
        seven_days_list.append(seven_days_income)
        fourteen_list.append(fourteen_days_income)
        index_list.append(index_status)

    df['one_day'] = day_after
    df['seven_days'] = seven_days_list
    
    df['fourteen_days'] = fourteen_list
    df['index_status'] = index_status

    df = df[(df['one_day'] >= -30) & (df['one_day'] <= 30) & (df['seven_days'] <= 70) & 
        (df['seven_days'] > -70) & (df['fourteen_days'] <= 85) & (df['fourteen_days'] > -80) &
        (df['index_status'] > -1)]


    cols = ['one_day', 'seven_days', 'fourteen_days']
    for i in range(len(cols)):
        df[result_cols[i]] = df[cols[i]].apply(lambda x : 1 if x > 0 else 0)
    
    return df

#box plot
def get_chart(df):

    fig, ax = plt.subplots(figsize = (16,8))
    sns.boxplot(data=df[['one_day','seven_days','fourteen_days']])
    plt.ylabel('변동률')
    plt.show()

#linear plot
def get_linear_chart(df):
    fig, ax = plt.subplots(figsize = (16,8))
    line_plot = sns.lineplot(data=df[['one_day','seven_days','fourteen_days']])
    line_plot.set(xticklabels=[])
    
    plt.ylabel('변동률')
    plt.legend(labels = ['다음 날 매도','7일 후 매도', '14일 후 매도'], loc='best', prop={'size': 15})
    plt.show()
    
def get_income_static(df):
    result = pd.DataFrame()
    for col in result_cols:
        tmp = pd.DataFrame(df[col].value_counts())
        result = pd.concat([result, tmp], axis=1)
    #columns 와 rows 변경    
    result = result.T
    result['percent'] = result[1] / (result[1] + result[0])
    
    result = result.fillna(1)
    return int(np.mean(result['percent']) * 100)
    

def get_correated_index(df):
    result_cols = ['one_day_trade_result', 'seven_trade_result', 'fourteen_trade_result']
    for col in result_cols:
        n = 0
        total = len(df[(df[col] == 1) & (df['index_status'] == 1)])
        if total > 0:
            effected = (df[1][n] - total) / df[1][n]
            # print(f'\n## {col}조건으로 검색된 종목과 코스피 지수와의 선형성을 띄는 종목의 비율은 {np.round(effected, 3)}입니다 ##')
            # print('')
        else:
            # print(f'\n## {col}조건으로 검색된 종목과 코스피 지수와의 선형성을 띄는 종목의 비율이 없습니다 - 즉 독립적 ##')
            print('')
        
def get_analyze(df):
    anal = df.describe()
    anal.drop(['one_day_trade_result', 'seven_trade_result', 'fourteen_trade_result', 'index_status'], axis = 1, inplace=True)
    # print(anal)
    #만약 123에서 50%값이 5%이상이면 손익량 20% 감소 75%가 5% 이상이면 10% 감소 기본값은 80%
    cols = ['one_day', 'seven_days', 'fourteen_days']

    sell_all = 0
    half_sell = 0
    last_sell = 0

    sell_all = np.round(anal['one_day'][3], 3)
    half_sell = np.round(anal['one_day'][5], 3)
    final_sell = np.round(anal['one_day'][6], 3)
    sell_div = 80

    for col in cols:
        n = 0
        if ((anal[col].iloc[5] - half_sell) > 5 & n == 1):
            sell_div -= 20
        elif ((anal[col].iloc[5] - final_sell) > 5 & n == 1):
            final_sell += ((final_sell - anal[col].iloc[5])/2)

        elif ((anal[col].iloc[5] - half_sell) > 5 & n == 2):
            sell_div -= 10

        elif ((anal[col].iloc[5] - final_sell) > 5 & n == 2):
            final_sell += ((final_sell - anal[col].iloc[5])/2)

        n +=1

    # if half_sell < 0 or final_sell < 0:
    #     print('수익이 나지 않는 조건식입니다. 다른 조건식의 종목들을 분석해주세요')
    #     print('')
    # else:
    #     print('-----조건식 종목 손익 진단-----')
    #     print('')

    msg = (f'손절라인은 {sell_all}%에 도달할 때 적정선으로 보이며 \n1차 손익가는 {half_sell}%에 도달할 때 보유량 대비 {sell_div}%를 매도\n남은 보유량은 손익가가 {final_sell}%에 도달할 때 전량 매도하는 것이 적절해보입니다.')
    # return msg
    return sell_all, half_sell, sell_div, final_sell

def get_total(df):
    get_chart(df)
    get_income_static(df)
    get_correated_index(df)
    get_analyze(df)








def file_format_download(request):
    path = request.GET['path']
    file_path = os.path.join(settings.MEDIA_ROOT, path)

    if os.path.exists(file_path):
        binary_file = open(file_path, 'rb')
        response = HttpResponse(binary_file.read(), content_type="application/octet-stream; charset=utf-8")
        response['Content-Disposition'] = 'attachment; filename=' + os.path.basename(file_path)
        return response
    else:
        message = '알 수 없는 오류가 발행하였습니다.'
        return HttpResponse("<script>alert('"+ message +"');history.back()'</script>")