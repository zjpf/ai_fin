# 上市状态: L上市 D退市 P暂停上市; 市场类别 (主板/创业板/科创板/CDR/北交所); 交易所 SSE上交所 SZSE深交所 BSE北交所; 是否沪深港通标的，N否 H沪股通 S深股通;
# 科创板 503 主板 2329 CDR 1 中小板 1015 创业板 1250 北交所 169
import tushare as ts
from datetime import datetime, timedelta
import pdb
import os
import csv
import time
import sys
import pandas as pd
csv.field_size_limit(sys.maxsize)
TOKEN_BUY = '1a312d7a80c6fcc1fd0a28116f8a1988b6756189d4db76e5bc603031'
TOKEN_BUY = '7972ea3bb966c5fc762122f3a2fcbe8b0baf22545528f6a18cfb5538'

class PullData:
    @staticmethod
    def get_file_name(step, dt=None):
        if dt is None:
            path = datetime.now().strftime('%Y_%m_%d')
        else:
            path = datetime.strptime(dt, '%Y%m%d').strftime('%Y_%m_%d')
        if step == 'stock_basic':
            return 'data/stock_basic.csv'  # 保存两份，历史版本加'_old'后缀
        elif step == 'daily_price':
            try:
                os.makedirs('data/daily_price')
            except:
                pass
            #if delta_dt is not None:
            #    path = (datetime.now() - timedelta(delta_dt)).strftime('%Y_%m_%d')
            return 'data/daily_price/{}.csv'.format(path)
            
    @staticmethod
    def get_stock_basic(is_pull=False):  # 1. 拉取最新股票列表，并返回股票信息词典；需要2000积分
        fields='ts_code,symbol,name,area,industry,market,exchange,curr_type,list_status,list_date,delist_date,is_hs,act_name,act_ent_type'
        ts.set_token(TOKEN_BUY)
        pro, stocks_fn, stock_fs = ts.pro_api(), PullData.get_file_name('stock_basic'), {}
        if is_pull:
            for list_status in ['L', 'D', 'P']:
                data = pro.stock_basic(list_status=list_status, fields=fields)
                data.to_csv(stocks_fn, mode='a', index=False)
        with open(stocks_fn, mode='r', encoding="utf-8") as fr:
            for fields in csv.reader(fr):
                fields = [v.replace('"', '') for v in fields]
                print(fields)
                if fields[5] in ['主板', '中小板']:
                    stock_fs[fields[0]] = fields[1:]
        return stock_fs

    @staticmethod
    def get_daily_price(si=0):  # 120积分 ToDo: 前复权的特殊处理
        dt_now = '20231029'  # datetime.now().strftime('%Y%m%d')  # 取最近10年的数据,主板和中小板
        dt_10y = '20130101'  # (datetime.now()-timedelta(365*10)).strftime('%Y%m%d')
        print(dt_10y, dt_now)
        stocks_fn, daily_price_fn, ri = PullData.get_file_name('stock_basic'), PullData.get_file_name('daily_price'), 0
        ts.set_token(TOKEN_BUY)
        with open(daily_price_fn, mode='a', encoding="utf-8") as f:
            with open(stocks_fn, mode='r', encoding="utf-8") as fr:
                for fields in csv.reader(fr):
                    fields = [v.replace('"', '') for v in fields]
                    print(fields)
                    ri += 1
                    if fields[5] not in ['主板', '中小板'] or fields[8] not in ['L'] or ri<si:
                        #print("skip_code: ", fields)
                        continue
                    print(dt_now, fields)
                    #pdb.set_trace()
                    code_events = {}
                    df = None
                    while df is None:
                        try:
                            df = ts.pro_bar(ts_code=fields[0], adj='qfq', start_date=dt_10y, end_date=dt_now, adjfactor=True)  # 前复权+复权因子
                        except:
                            time.sleep(60)
                            df = None
                    if df is None:
                        print("None_code: ", fields)
                        continue
                    for i, row in df.iterrows():
                        code = row['ts_code']
                        event = [row['trade_date'], row['high'], row['low'], row['open'], row['close'], row['vol'], row['amount'], row['adj_factor']]
                        od = code_events.setdefault(code, [])
                        od.append(event)
                    #pdb.set_trace()
                    for code, events in code_events.items():
                        sorted_events = sorted(events, key=lambda x: x[0])
                        f.write("""{},"{}",{}\n""".format(code, sorted_events, ','.join(fields[1:])))
                    time.sleep(5)
        print("Done 20231029 crawl 3184主板")

    @staticmethod
    def get_qfq(stock_fs, f, dt_now, dt_10y='20130101'):
        for code, fs in stock_fs.items():
            print(code, fs)
            code_events = {}
            df = None
            while df is None:
                try:
                    df = ts.pro_bar(ts_code=code, adj='qfq', start_date=dt_10y, end_date=dt_now, adjfactor=True)  # 前复权+复权因子
                except:
                    time.sleep(60)
                    df = None
            if df is None:
                print("None_code: ", fields)
                continue
            for i, row in df.iterrows():
                code = row['ts_code']
                event = [row['trade_date'], row['high'], row['low'], row['open'], row['close'], row['vol'], row['amount'], row['adj_factor']]
                od = code_events.setdefault(code, [])
                od.append(event)
            #pdb.set_trace()
            for code, events in code_events.items():
                sorted_events = sorted(events, key=lambda x: x[0])
                f.write("""{},"{}",{}\n""".format(code, sorted_events, ','.join(fs)))
                f.flush()
            time.sleep(5)
        
    @staticmethod
    def merge_daily_price(m_dt, si=0):
        code_event = {}  # 历史数据
        with open(PullData.get_file_name('daily_price', m_dt), mode='r', encoding="utf-8") as fr:
            for row in csv.reader(fr):
                code_event[row[0]] = row[1:]
        # 当天增量数据
        dt_now = datetime.now().strftime('%Y%m%d')  # 取最近10年的数据,主板和中小板
        dt_inc = (datetime.strptime(m_dt, '%Y%m%d')+timedelta(1)).strftime('%Y%m%d')
        print(dt_inc, dt_now)
        stocks_fn, daily_price_fn, ri = PullData.get_file_name('stock_basic'), PullData.get_file_name('daily_price'), 0
        #pdb.set_trace()
        ts.set_token(TOKEN_BUY)
        with open(daily_price_fn, mode='a', encoding="utf-8") as f:
            with open(stocks_fn, mode='r', encoding="utf-8") as fr:
                for fields in csv.reader(fr):
                    fields = [v.replace('"', '') for v in fields]
                    #print(fields)
                    ri += 1
                    if fields[5] not in ['主板', '中小板'] or fields[8] not in ['L'] or ri<si:
                        continue
                    print(dt_now, fields)
                    #pdb.set_trace()
                    code_events, df = {}, None
                    while df is None:
                        try:
                            df = ts.pro_bar(ts_code=fields[0], adj='qfq', start_date=dt_inc, end_date=dt_now, adjfactor=True)  # 前复权+复权因子
                        except:
                            time.sleep(3)
                            df = None
                    for i, row in df.iterrows():
                        code = row['ts_code']
                        event = [row['trade_date'], row['high'], row['low'], row['open'], row['close'], row['vol'], row['amount'], row['adj_factor']]
                        od = code_events.setdefault(code, [])
                        od.append(event)
                    #pdb.set_trace()
                    # 合并历史
                    for code, events in code_events.items():
                        if fields[0] not in code_event:
                            print("Not cover", fields[0])
                            continue
                        sorted_events = sorted(events, key=lambda x: x[0])
                        last_event = eval(code_event[fields[0]][0])
                        adj_ratio = last_event[-1][-1]/sorted_events[-1][-1]
                        for e in sorted_events:
                            if adj_ratio == 1.0:
                                last_event.append(e)
                            else:
                                last_event.append([fi*adj_ratio if 1<=i<=4 else fi for i, fi in enumerate(e)])
                        f.write("""{},"{}",{}\n""".format(code, last_event, ','.join(fields[1:])))
                    f.flush()
                    #pdb.set_trace()
                    time.sleep(0.12)
        print("Done")

    @staticmethod
    def merge_daily_price_v2(m_dt, inc_dt, si=0):
        stock_fs = PullData.get_stock_basic()
        code_event = {}  # 历史数据
        with open(PullData.get_file_name('daily_price', m_dt), mode='r', encoding="utf-8") as fr:
            for row in csv.reader(fr):
                code_event[row[0]] = row[1:]
        # 当天增量数据
        stocks_fn, daily_price_fn, ri = PullData.get_file_name('stock_basic'), PullData.get_file_name('daily_price'), 0
        ts.set_token(TOKEN_BUY)
        with open(daily_price_fn, mode='w', encoding="utf-8") as f:
            #PullData.get_qfq(stock_fs, f, m_dt)

            dfs_1, dfs_2, pro = [], [], ts.pro_api()
            for dt in inc_dt:
                print(dt)
                #pdb.set_trace()
                df2, df = None, None
                while df is None:
                    try:
                        df = pro.daily(trade_date=dt)
                    except:
                        time.sleep(3)
                        df = None
                while df2 is None:
                    try:
                        df2 = pro.query('adj_factor',  trade_date=dt)
                    except:
                        time.sleep(3)
                        df2 = None
                if df.size > 0:
                    dfs_1.append(df)
                #pdb.set_trace()
                if len(df2) > 0:
                    dfs_2.append(df2)
            #pdb.set_trace()
            df_s1, df_s2, code_events = pd.concat(dfs_1), pd.concat(dfs_2), {}
            df = df_s1.merge(df_s2, how='left', on=['ts_code', 'trade_date'] )
            #pdb.set_trace()
            for i, row in df.iterrows():
                code = row['ts_code']
                event = [row['trade_date'], row['high'], row['low'], row['open'], row['close'], row['vol'], row['amount'], row['adj_factor']]
                od = code_events.setdefault(code, [])
                od.append(event)
            #pdb.set_trace()
            # 合并历史: TODO: 不支持新股；修复复权因子变动bug；
            qfq_stocks = {}
            for code in code_events.keys()&code_event.keys():
                events = code_events.get(code, [])
                sorted_events = sorted(events, key=lambda x: x[0])
                if code not in code_event:
                    if code in stock_fs:
                        print(stock_fs.get(code, '主板/中小板新上'))
                        f.write("""{},"{}",{}\n""".format(code, sorted_events, ','.join(stock_fs[code])))
                        f.flush()
                    else:
                        print("Not cover: ", code, ", ", stock_fs.get(code, 'Not in 主板/中小板 stocks'))
                    continue

                last_event = eval(code_event[code][0])
                adj_ratio = last_event[-1][-1]/sorted_events[-1][-1]
                if adj_ratio == 1.0:
                    for e in sorted_events:
                        last_event.append(e)
                    f.write("""{},"{}",{}\n""".format(code, last_event, ','.join(code_event[code][1:])))
                    f.flush()
                else:
                    #pdb.set_trace()
                    qfq_stocks[code] = code_event[code][1:]
            print(qfq_stocks)

            #pdb.set_trace()
            time.sleep(0.12)
            PullData.get_qfq(qfq_stocks, f, inc_dt[-1])
        print("Done")


if __name__ == "__main__":
    import sys
    # PullData.get_stock_basic()
    #PullData.merge_daily_price_v2('20231029', inc_dt=['20231029', '20231030', '20231031', '20231101', '20231102', '20231103']+
    #  ['20231106', '20231107','20231108','20231109','20231110'])
    #dts = map(str, list(range(20231121, 20231131))+list(range(20231201, 20231226)))
    ct, et, dts = datetime.strptime(sys.argv[1], '%Y%m%d'), datetime.strptime(sys.argv[2], '%Y%m%d'), []
    st = ct + timedelta(1)
    while st <= et:
        #print(st, et)
        dts.append(st.strftime('%Y%m%d'))
        st += timedelta(1)
    #dts = ['20240412']
    print(dts)
    PullData.merge_daily_price_v2(sys.argv[1], inc_dt=dts)
    #PullData.merge_daily_price('20231029', si=510+296+1)
    #PullData.get_daily_price(si=510+184+1+12)

