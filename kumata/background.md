# コンテスト背景
## データの概要
### 種類
**人材**・**旅行**・**不動産**・**アパレル** の計4業種におけるユーザーの行動履歴データ

### 期間
* 学習データ: 4/1～4/30の **１か月間**
* 予測対象データ 5/1～5/7の **１週間**

### 予測対象ユーザー
4月中に行動履歴の存在するユーザー

### ユーザーが商品に対してとる行動
####  商品の購入(cv)  
* event_type == 3

####  広告をクリック(cl)
* event_type == 2

####  商品の詳細ページを閲覧(pd)
* event_type == 1

####  商品をカートに入れる(ca)
* event_type == 0

## 評価方法
![image](https://i.deepanalytics.jp/i/wh0t4i3541)

予測精度の評価は、nDCG(normalized discounted cumulative gain)を使用します（右図参照）。この値は、モデルの性能が良いほど大きくなり、1に近くなります。関連度（relevance）はcv（コンバージョン）を3、cl（広告をクリック）を2、pd（商品ページ閲覧）を1、それ以外は0とします。ただしコンバージョンは広告経由のみ評価対象とします。クエリごとの最大推薦数kは22とします。予測値の出力形式についてはダウンロードページの応募用サンプルファイルをご参照ください。また、test.tsvに記載されているすべてのユーザーに対して予測を行ってください。

## ルール
#### １ユーザにつき１アカウント
* コンテスト参加者は1人につき１アカウントまでです。

#### 他参加者との情報共有は禁止
* コンテスト参加者が同じチーム以外の参加者と本コンテストの予測に関連するデータ・ソースコードを共有する行為は禁止です。

#### 学習データは提供データのみ
*  配布する学習データ以外のデータを用いてモデルを学習することは禁止です。

#### オープン且つ無料なツールのみ
* モデルの学習に利用するツールは、オープン且つ無料なもの（python, R 等）に限定します。

#### 業種ごとに個別にモデリングを行うこと
*  学習、予測ロジックがすべての業種で共通するのは問題ありませんが、ある1つの業種について学習と予測を行う際に、
*  他の業種のデータを使わないと動かないようなモデリングを行うことは禁止とします。

#### 汎用的なモデリングであること
* 提案した方法が一般的な環境において追加費用負担を伴わず、再現及び継続使用可能であることを保証する必要があります。
* 同じフォーマットで、異なるデータを入力した場合にも同様なロジックで予測できなければなりません。
* 例えば、根拠無く局所的に予測値を修正することは禁止となります。また、ユーザーIDや商品IDの文字の並び方などを学習させ、予測に利用することも
* 禁止となります。
* （基準について不安がある場合は、事務局までお問い合わせください）
