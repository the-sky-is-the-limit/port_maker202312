import streamlit as st
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import math
import numpy as np
import plotly.graph_objects as go
import datetime
import matplotlib.ticker as mtick
import seaborn as sns
japanize_matplotlib.japanize()
import io
import xlsxwriter
from matplotlib.ticker import FuncFormatter
from scipy import stats



# 日本語フォントの設定
japanize_matplotlib.japanize()

# to_percentage: 値をパーセンテージ形式に変換する関数
def to_percentage(value):
    if isinstance(value, str):
        return value
    else:
        return f"{round(value * 100, 2)}%"

# to_percentage2: 値をパーセンテージ形式に変換する別の関数（小数点以下の桁数が異なる）
def to_percentage2(value):
    if isinstance(value, str):
        return value
    else:
        return f"{round(value, 2)}%"

# quants_data: ポートフォリオの量的分析を行う関数
def quants_data(df, index_name):
    def sortino_ratio(returns_series, target_return=0):
        downside_returns = returns_series.copy()
        downside_returns[downside_returns > target_return] = 0
        sortino = (returns_series.mean() - target_return) / downside_returns.std()
        return sortino * np.sqrt(12)

    quants_datas = pd.DataFrame(columns=df.columns, index=[
        '1年リターン', '3年平均リターン', '5年平均リターン', '10年平均リターン', '15年平均リターン', '20年平均リターン',
        'Since Return', 'Since Risk', 'sharpe_ratio', 'sortino_ratio', 'Max_DrowDawn', '最大下落月', '月次勝率',
        '月次最大上昇率', '月次最大下落率', '平均月次リターン', index_name + 'との相関性', 'データ開始', 'データ終了', '運用期間'
    ])


    # Check df size before iloc access
    df_size = len(df)
    if df_size >= 13:
        quants_datas.loc['1年リターン',:] = df.iloc[-1]/df.iloc[-13]-1
    else:
        quants_datas.loc['1年リターン',:] = np.nan
    if df_size >= 37:
        quants_datas.loc['3年平均リターン',:] = (df.iloc[-1]/df.iloc[-37])**(12/36)-1
    else:
        quants_datas.loc['3年平均リターン',:] = np.nan
    if df_size >= 61:
        quants_datas.loc['5年平均リターン',:] = (df.iloc[-1]/df.iloc[-61])**(12/60)-1
    else:
        quants_datas.loc['5年平均リターン',:] = np.nan
    if df_size >= 121:
        quants_datas.loc['10年平均リターン',:] = (df.iloc[-1]/df.iloc[-121])**(12/120)-1
    else:
        quants_datas.loc['10年平均リターン',:] = np.nan
    if df_size >= 181:
        quants_datas.loc['15年平均リターン',:] = (df.iloc[-1]/df.iloc[-181])**(12/180)-1
    else:
        quants_datas.loc['15年平均リターン',:] = np.nan
    if df_size >= 241:
        quants_datas.loc['20年平均リターン',:] = (df.iloc[-1]/df.iloc[-241])**(12/240)-1
    else:
        quants_datas.loc['20年平均リターン',:] = np.nan

    # 設定来リターン/リスク
    for x in df.columns.values:
        x_len = len(df[x].dropna().index)-1
        quants_datas.loc['Since Return',f'{x}']=(df[f'{x}'].dropna().iloc[-1]/df[f'{x}'].dropna().iloc[0])**(12/x_len)-1
        quants_datas.loc['Since Risk',f'{x}']=df[f'{x}'].pct_change().dropna().std()*(12**(1/2))
        quants_datas.loc['データ開始',f'{x}']=df[f'{x}'].dropna().index[0]
        quants_datas.loc['データ終了',f'{x}']=df[f'{x}'].dropna().index[-1]
        quants_datas.loc['運用期間',f'{x}']=len(df[f'{x}'].dropna())
        quants_datas.loc['sharpe_ratio',f'{x}'] = ((df[f'{x}'].dropna().iloc[-1]/df[f'{x}'].dropna().iloc[0])**(12/x_len)-1) / (df[f'{x}'].pct_change().dropna().std()*(12**(1/2)))

        # ソルティノレシオの計算
        returns_series = df[f'{x}'].pct_change().dropna()
        quants_datas.loc['sortino_ratio', f'{x}'] = sortino_ratio(returns_series)

        # 世界株との相関性の計算（'世界株'を選択されたインデックスに変更）
        for x in df.columns.values:
            if x != index_name:
                quants_datas.loc[index_name + 'との相関性', x] = df[x].pct_change().dropna().corr(df[index_name].pct_change().dropna()).round(2)
            else:
                quants_datas.loc[index_name + 'との相関性', x] = 1  # インデックスと自身の相関性は1

    # 最大ドローダウン
    quants_datas.loc['Max_DrowDawn',:]=(df/df.cummax()-1).min()
    quants_datas.loc['最大下落月',:]=(df/df.cummax()-1).idxmin()

    # ポジティブ月
    df_rtn=df.pct_change()
    quants_datas.loc['月次勝率',:]=(df_rtn[df_rtn > 0].count())/df_rtn.count()

    #月次最大上昇率
    quants_datas.loc['月次最大上昇率',:]=df_rtn.max()

    #月次最大下落率
    quants_datas.loc['月次最大下落率',:]=df_rtn.min()

    #平均月次リターン
    quants_datas.loc['平均月次リターン',:]=df_rtn.mean()

    # 変更する行を指定し、日付を年-月-日表示に変更
    for row in ['最大下落月', 'データ開始', 'データ終了']:
        quants_datas.loc[row] = pd.to_datetime(quants_datas.loc[row]).dt.strftime('%Y-%m')

    # to_percentage関数を適用する行を指定し、パーセンテージ形式に変換
    for row in ['1年リターン', '3年平均リターン', '5年平均リターン', '10年平均リターン', '15年平均リターン', '20年平均リターン', 'Since Return', 'Since Risk', 'Max_DrowDawn', '月次勝率', '月次最大上昇率', '月次最大下落率', '平均月次リターン']:
        quants_datas.loc[row] = quants_datas.loc[row].apply(lambda x: to_percentage(x) if pd.notnull(x) else x)

    for row in ['sharpe_ratio', 'sortino_ratio']:
        quants_datas.loc[row] = quants_datas.loc[row].apply(lambda x: to_percentage2(x) if pd.notnull(x) else x)

    quants_datas = quants_datas.dropna(how='all')

    return quants_datas


# chart_compare: ポートフォリオとインデックスを比較するチャートを生成する関数
def chart_compare(prices_df, pf_name, index_name):
    sns.set()
    japanize_matplotlib.japanize()
    sns.set_context(context='talk', font_scale=1)
    fig, ax = plt.subplots(figsize=(11,6))
    sns.lineplot(x=prices_df.index, y=pf_name, data=prices_df, ax=ax, linewidth=4, label=pf_name)
    sns.lineplot(x=prices_df.index, y=index_name, data=prices_df, linestyle='dashed', ax=ax, linewidth=3, label=index_name)

    # get last price for portfolio and "世界株"
    last_price_pf = prices_df[pf_name].iloc[-1]
    last_price_world = prices_df[index_name].iloc[-1]

    # add annotations for last price
    ax.annotate(f'{last_price_pf:,.0f}', 
                xy=(prices_df.index[-1], last_price_pf), 
                xytext=(15, 15), 
                textcoords='offset points',
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->'))
    
    ax.annotate(f'{last_price_world:,.0f}', 
                xy=(prices_df.index[-1], last_price_world), 
                xytext=(15, 15), 
                textcoords='offset points',
                arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->'))
    
    # Set graph's title, axis labels etc. (optional)
    ax.set_title(f'{pf_name}と{index_name}の月次株価')
    ax.set_xlabel('日付')
    ax.set_ylabel('株価')
    
    ax.legend()
    st.pyplot(fig)

# scatter: 月次リターンの散布図を生成する関数
def scatter(two_prices_df, index_name):
    sns.set(font='IPAexGothic', context='notebook')
    num_plots = len(two_prices_df.columns)

    # サブプロットの数とレイアウト
    num_rows = (num_plots + 1) // 2
    num_cols = min(num_plots, 2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(11, 6 * num_rows))
    axes = axes.flatten()

    for i, x in enumerate(two_prices_df.columns):
        data = two_prices_df.dropna().pct_change()

        ax = axes[i]
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))
        ax.set_ylim(-0.25, 0.25)

        sns.scatterplot(data=data[f'{x}'], ax=ax)

        # 平均リターンの線と数字のアノテーション
        mean_return = data[f'{x}'].mean()
        ax.axhline(y=mean_return, linestyle='--', color='r', label='平均リターン')
        ax.annotate(f'{mean_return:.2%}', xy=(data[f'{x}'].index[-1], mean_return), xycoords='data', va='center', ha='left', fontsize=9, backgroundcolor='w')

        ax.set_title(f"{x}：月次リターン分布")

    # 余分なサブプロットを非表示にする
    if num_plots < len(axes):
        for j in range(num_plots, len(axes)):
            axes[j].axis('off')

    plt.tight_layout()
    st.pyplot(fig) 

# bar: 年次リターンのバーチャートを生成する関数
def bar(price, index_name):
    temp = price.dropna() / price.dropna().iloc[0] * 100
    temp_resample = temp.resample('Y').last().pct_change()
    temp_resample.iloc[0] = temp.resample('Y').last().iloc[0] / 100 - 1
    temp_resample = temp_resample.stack().rename_axis(['Date', 'Asset']).reset_index().rename(columns={0: "Return"})

    temp_resample['Year-Month'] = temp_resample['Date'].dt.to_period('M')

    fig, ax = plt.subplots(figsize=(13, 6.5), facecolor="w")
    ax = sns.barplot(data=temp_resample, x="Year-Month", y="Return", hue="Asset")
    plt.title('Annual Returns', fontsize=13)
    plt.xticks(rotation=45)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

    # Add annotations (xx%) to each bar
    for p in ax.patches:
        if p.get_height() >= 0:
            ax.annotate(format(p.get_height(), '.0%'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='heavy',
                        color='black')
        else:
            ax.annotate(format(p.get_height(), '.0%'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center',
                        va='top',
                        xytext=(0, -9),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='heavy',
                        color='black')

    # Adjust the height of the graph
    bottom, top = plt.ylim()
    plt.ylim(bottom - 0.05, top + 0.05)

    plt.legend()
    st.pyplot(fig)


#マイナスの年だけ棒グラフ表示
def bar_minus(price, index):
    temp = price.dropna() / price.dropna().iloc[0] * 100
    temp_resample = temp.resample('Y').last().pct_change()
    temp_resample.iloc[0] = temp.resample('Y').last().iloc[0] / 100 - 1
    temp_resample = temp_resample.stack().rename_axis(['Date', 'Asset']).reset_index().rename(columns={0: "Return"})

    index_resample = index.resample('Y').last().pct_change()
    index_resample.iloc[0] = index.resample('Y').last().iloc[0] / 100 - 1
    negative_years = index_resample[index_resample.iloc[:, 0] < 0].index.to_period('Y').tolist()

    temp_resample['Year-Month'] = temp_resample['Date'].dt.to_period('Y')
    temp_resample = temp_resample[temp_resample['Year-Month'].isin(negative_years)]

    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor="w")  # 修正した行
    ax = sns.barplot(data=temp_resample, x="Year-Month", y="Return", hue="Asset")
    plt.title('年次騰落率（マイナスの年）', fontsize=13)
    plt.xticks(rotation=45)

    # y軸パーセンテージ表示
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

    # 各barにannotation(xx%)を追加する
    for p in ax.patches:
        if p.get_height() >= 0:
            ax.annotate(format(p.get_height(), '.0%'),
                        (p.get_x() + p.get_width() / 2.,
                         p.get_height()),
                        ha='center',
                        va='center',
                        xytext=(0, 9),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='heavy',
                        color='black')
        else:
            ax.annotate(format(p.get_height(), '.0%'),
                        (p.get_x() + p.get_width() / 2.,
                         p.get_height()),
                        ha='center',
                        va='top',
                        xytext=(0, -9),
                        textcoords='offset points',
                        fontsize=10,
                        fontweight='heavy',
                        color='black')

    # グラフの高さを調整
    bottom, top = plt.ylim()
    plt.ylim(bottom - 0.05, top + 0.05)

    plt.legend()
    st.pyplot(fig)


# dd4: ドローダウンチャートを生成する関数
def dd4(hedgefund, index, index_name):
    def DD(fund_name, ax1, ax2):
        DD_hedgefund = pd.DataFrame()
        hist_nav = fund_name
        log_ret = np.log(hist_nav / hist_nav.shift())
        DD_hedgefund['cumret'] = log_ret.cumsum().apply(np.exp) * 100
        DD_hedgefund['cummax'] = DD_hedgefund['cumret'].cummax()
        DD_hedgefund['DD'] = (DD_hedgefund['cumret'] - DD_hedgefund['cummax']) / DD_hedgefund['cummax']

        ax1.plot(DD_hedgefund['cummax'], color='Red', linestyle='--')
        ax1.plot(DD_hedgefund['cumret'], color='Blue')
        ax1.set_title(f'{hist_nav.name}チャート＆高値', c="darkred", size="large")
        ax1.set_ylabel("基準価格")

        ax2.plot(DD_hedgefund['DD'], label=hist_nav.name)
        ax2.set_title(f'ドローダウン', c="darkred", size="large")
        ax2.set_ylabel("下落率")
        ax2.set_ylim(-0.55, 0.05)  # y軸の範囲を指定
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

    sns.set(font='IPAexGothic', context='notebook')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(11, 6))

    # 余白を設定
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    DD(hedgefund, ax1, ax3)
    DD(index, ax2, ax4)

    # 各プロットに凡例を追加
    ax3.legend()
    ax4.legend()
    st.pyplot(fig)

# calculate_pct_change: 運用期間別のリターンを計算する関数
def calculate_pct_change(dataframe):
    column_name = dataframe.columns[0]
    po = pd.DataFrame()
    po['1年'] = dataframe.pct_change(12).sort_values(by=column_name, ascending=False).reset_index(drop=True)
    po['3年'] = dataframe.pct_change(36).sort_values(by=column_name, ascending=False).reset_index(drop=True)
    po['5年'] = dataframe.pct_change(60).sort_values(by=column_name, ascending=False).reset_index(drop=True)
    po['7年'] = dataframe.pct_change(84).sort_values(by=column_name, ascending=False).reset_index(drop=True)
    po['10年'] = dataframe.pct_change(120).sort_values(by=column_name, ascending=False).reset_index(drop=True)
    return (po*100).round(2)

def calculate_max_min(df):
    acc = pd.DataFrame()

    acc['1年最大'] = df.dropna().pct_change(12).max()
    acc['1年最小'] = df.dropna().pct_change(12).min()
    acc['3年最大'] = df.dropna().pct_change(36).max()
    acc['3年最小'] = df.dropna().pct_change(36).min()
    acc['5年最大'] = df.dropna().pct_change(60).max()
    acc['5年最小'] = df.dropna().pct_change(60).min()

    return ((acc.T)*100).round(2) #.applymap('{:.1%}'.format)

# nan_to_blank: NaN値を空白に変換する関数
def nan_to_blank(value):
    if np.isnan(value):
        return ''
    return value

def table_return(df):
    # 指数化
    df_index = df / df.iloc[0] * 100

    # 年単位に変換
    df_index_y = df_index.resample('Y').last()

    # 年率のリターン実績計算
    df_index_y_pct = df_index_y.pct_change()

    # 最初の年が空欄になるので、修正
    df_index_y_pct.iloc[0] = df_index_y.iloc[0] / 100 - 1

    # 月次リターン
    df_ret = df.pct_change().dropna()

    # 月次リターンをピボットでクロス集計
    monthly_return = pd.pivot_table(df_ret, index=df_ret.index.year, columns=df_ret.index.month)
    monthly_return.columns = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']

    # 年次リターンの行数が月次リターンの行数より多い場合、最初の年を削除
    if len(df_index_y_pct) > len(monthly_return):
        df_index_y_pct = df_index_y_pct.iloc[1:]

    # df_index_y_pctのインデックスをmonthly_returnのインデックスに合わせる
    try:
        df_index_y_pct = df_index_y_pct.set_index(monthly_return.index)
    except ValueError as e:
        print(f"Error setting index: {e}")
        # さらなるエラーハンドリングや修正処理をここに追加

    # 表のNaNを空欄に変換
    result = pd.concat([monthly_return, df_index_y_pct], axis=1)
    result = result.applymap(nan_to_blank)

    return result

def one_year_corr(df, pf_name, main_index, selected_index):
    # 騰落率の計算
    corr_hedge = df.dropna().pct_change().dropna()
    
    # 1年間のローリング相関を計算
    corr_hedge[f'{main_index}との1年相関性'] = corr_hedge[pf_name].rolling(12).corr(corr_hedge[selected_index])
    
    # グラフの設定
    sns.set(font='IPAexGothic', context='notebook')
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.ylim(-1, 1)
    
    # 相関性のラインプロット
    sns.lineplot(data=corr_hedge[f'{main_index}との1年相関性'], linewidth=3)
    
    # タイトルの設定
    plt.title(f"{pf_name}と{selected_index}の1年相関性")
    
    # Streamlitでグラフを表示
    st.pyplot(fig)

def histograms(price):
    sns.set_theme(font='IPAexGothic', context='talk')

    for i in price.columns:
        fig, ax = plt.subplots(figsize=(16, 4))

        # ヒストグラムを描画
        data = price[i].pct_change()
        hist_data = sns.histplot(data=data,
                                 binrange=(data.min(), data.max()),
                                 bins=16,
                                 shrink=0.8,
                                 ax=ax)

        ax.set_title(f"{i}_騰落率", pad=20)  # padパラメータでタイトルの位置を調整
        
        ax.set_ylabel("月数")
        plt.xlim(-0.20, 0.20)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=None, symbol='%', is_latex=False))

        # 各バーの上にアノテーションを追加
        for p in hist_data.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', 
                        va='bottom', 
                        xytext=(0, 5),
                        textcoords='offset points')

        # Streamlitでグラフを表示
        st.pyplot(fig)

def plot_annualized_return_distribution(df_price):
    # 年率のリスクとリターンを計算
    sns.set_theme(font='IPAexGothic', context='notebook')
    monthly_returns = df_price.pct_change().dropna()
    annual_return =(df_price.dropna().iloc[-1]/df_price.dropna().iloc[0])**(12/x_len)-1
    # annual_return = (monthly_returns.mean() + 1) ** 12 - 1
    annual_std = monthly_returns.std() * np.sqrt(12)

    # DataFrame の値をスカラーに変換
    annual_return = annual_return.item()
    annual_std = annual_std.item()

    # 正規分布のグラフを描画
    x = np.linspace(annual_return - 3*annual_std, annual_return + 3*annual_std, 1000)
    y = stats.norm.pdf(x, annual_return, annual_std)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, 'gray', lw=2)

    # 2σの範囲を色塗り
    ax.fill_between(x, y, where=((x < annual_return + 2*annual_std) & (x > annual_return - 2*annual_std)), color='skyblue', label='2σ Range')

    # 1σの範囲を色塗り
    ax.fill_between(x, y, where=((x < annual_return + annual_std) & (x > annual_return - annual_std)), color='yellow', label='1σ Range')

    # 平均リターンの位置に縦の中央線を追加
    ax.axvline(x=annual_return, color='r', linestyle='--', label='Mean Return')

    # 1σ、2σ、および平均リターンの位置にテキストを示す
    plt.text(annual_return, -0.02, f'{annual_return*100:.2f}%', horizontalalignment='center', verticalalignment='top', color='black')
    plt.text(annual_return - annual_std, -0.02, f'{(annual_return - annual_std)*100:.2f}%', horizontalalignment='center', verticalalignment='top',color='black')
    plt.text(annual_return + annual_std, -0.02, f'{(annual_return + annual_std)*100:.2f}%', horizontalalignment='center', verticalalignment='top',color='black')
    plt.text(annual_return - 2*annual_std, -0.02, f'{(annual_return - 2*annual_std)*100:.2f}%', horizontalalignment='center', verticalalignment='top',color='black')
    plt.text(annual_return + 2*annual_std, -0.02, f'{(annual_return + 2*annual_std)*100:.2f}%', horizontalalignment='center', verticalalignment='top',color='black')

    # 指定されたパーセンタイル値を追加
    plt.text(annual_return - 2.5*annual_std, 0, "2.5%", horizontalalignment='center', verticalalignment='bottom', color='blue')
    plt.text(annual_return - 1.5*annual_std, 0.3, "13.5%", horizontalalignment='center', verticalalignment='bottom', color='blue')
    plt.text(annual_return - 0.5*annual_std, 1, "34%", horizontalalignment='center', verticalalignment='bottom', color='blue')
    plt.text(annual_return + 0.5*annual_std, 1, "34%", horizontalalignment='center', verticalalignment='bottom', color='blue')
    plt.text(annual_return + 1.5*annual_std, 0.3, "13.5%", horizontalalignment='center', verticalalignment='bottom', color='blue')
    plt.text(annual_return + 2.5*annual_std, 0, "2.5%", horizontalalignment='center', verticalalignment='bottom', color='blue')

    # グラフのx軸にカスタムフォーマッターを適用
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{100*x:.2f}%"))

    ax.set_title(f"{df_price.columns[0]}の統計的なリターンの分布予想図")
    ax.set_xlabel("Returns")
    ax.set_ylabel("Probability Density")
    ax.legend()

    ax.grid(True)

    # Streamlit でグラフを表示
    st.pyplot(fig)

    


#年次リターン計算
def data_index(df):
  df_y=df.resample('Y').last().pct_change()
  df_y.iloc[0]=df.resample('Y').last().iloc[0]/100-1
  return df_y


# タイトル設定
st.title('Portfolio Calculation')

# サイドバーでExcelファイルのアップロードを許可
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=['xlsx'])



if uploaded_file is not None:
    data = pd.read_excel(uploaded_file, index_col=0, parse_dates=True)
    stocks = list(data.columns)

    selected_stocks = st.sidebar.multiselect('Select portfolio stocks', stocks)
    selected_index = st.sidebar.selectbox('Select an index for comparison', stocks)
    pf_name = st.sidebar.text_input("Enter portfolio name", "PF")

    # インデックスがポートフォリオ銘柄に含まれているかチェック
    index_in_stocks = selected_index in selected_stocks

    if selected_stocks and selected_index:
        investment_amount = []

        # すべての選択された銘柄の投資額を収集
        for stock in selected_stocks:
            amount = st.sidebar.number_input(f'Set your investment amount for {stock}', min_value=0, value=1, key=f'{stock}_amount')
            investment_amount.append(amount)

        # インデックスがポートフォリオ銘柄の一部であった場合、その銘柄の投資額を追加
        if index_in_stocks:
            amount_index = st.sidebar.number_input(f'Set your investment amount for {selected_index}', min_value=0, value=1, key=f'{selected_index}_amount')
            investment_amount.append(amount_index)
        else:
            # インデックスがポートフォリオ銘柄の一部ではない場合は、投資額リストに0を追加
            investment_amount.append(0)

        investment_weight = np.array(investment_amount) / sum(investment_amount)

        select_data = data[selected_stocks + [selected_index]].dropna()
        select_data = select_data / select_data.iloc[0] * 100

        # 選択された銘柄（インデックス含む）でポートフォリオを計算
        select_data[pf_name] = select_data[selected_stocks + [selected_index]] @ investment_weight



        df = select_data[[pf_name, selected_index]]
        df2 = df.copy().round(2)
        df2.index = pd.to_datetime(df2.index).strftime('%Y-%m-%d')

        st.dataframe(df2)

        quants_result = quants_data(df, selected_index)
        st.markdown('<p style="background-color: black; color: white; padding: 8px;">Basic Data (%))</p>', unsafe_allow_html=True)
        st.table(quants_result)

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">Comparison Chart</p>', unsafe_allow_html=True)
        chart_compare(df, pf_name, selected_index)

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">Scatter Chart</p>', unsafe_allow_html=True)
        scatter(df, selected_index)

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">Annual Return</p>', unsafe_allow_html=True)
        bar(df, selected_index)

        st.markdown(f'<p style="background-color: black; color: white; padding: 8px;">Annual Return (インデックスがマイナスの年)</p>', unsafe_allow_html=True)
        bar_minus(df, df[[selected_index]])

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">DrawDown Chart</p>', unsafe_allow_html=True)
        dd4(df[pf_name], df[selected_index], selected_index)

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">運用期間別損益 (%)</p>', unsafe_allow_html=True)
        pf = calculate_pct_change(df[[pf_name]])
        st.write(pf.dropna(how='all'))

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">期間別最大値最小値 (%)</p>', unsafe_allow_html=True)
        cal = calculate_max_min(df)
        st.write(cal)
        tr = table_return(df[[pf_name]])

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">１年相関性</p>', unsafe_allow_html=True)
        one_year_corr(df, pf_name, selected_index, selected_index)

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">ヒストグラム</p>', unsafe_allow_html=True)
        histograms(df)

        st.markdown('<p style="background-color: black; color: white; padding: 8px;">正規分布</p>', unsafe_allow_html=True)
        plot_annualized_return_distribution(df[[pf_name]])
        plot_annualized_return_distribution(df[[selected_index]])


        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='df')
            quants_result.to_excel(writer, sheet_name='quants_result')
            pf.to_excel(writer, sheet_name='期間損益一覧')
            cal.to_excel(writer, sheet_name='期間別最大最小')
            tr.to_excel(writer, sheet_name='月次リターン表')
            data_index(select_data).to_excel(writer, sheet_name='年次リターン')
            select_data.to_excel(writer, sheet_name='select_data')
            quants_data(select_data, selected_index).to_excel(writer, sheet_name='quants_data')

        output.seek(0)

        st.download_button(
            label="Download data as Excel",
            data=output,
            file_name=f'{pf_name}data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.write("No stocks or index selected")
else:
    st.write("Please upload a file")
