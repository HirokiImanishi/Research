# -*- coding: utf-8 -*-
# https://github.com/dandelin/watcha-doc2vec-regression 参考
# http://qiita.com/Leonhalt2714/items/dc704f841d21627e988a

import codecs
import os
import sys
from natto import MeCab
import collections
#import doc2vec
from gensim import models
import numpy as np
from gensim.models.doc2vec import LabeledSentence
from gensim.models import word2vec
from sklearn.svm import SVR
#import sklearn.svm as svm
import linecache
from sklearn.metrics import mean_squared_error
from math import sqrt

#それぞれ2014,15年度の四季報のテキストのフォルダ
INPUT_DOC_DIR = './3roa-createOneInputFiledata/doc2014'
INPUT_DOC_DIR2 = './3roa-createOneInputFiledata/doc2015'
#学習済みのdoc2vecのモデルを読み込む
model = models.Doc2Vec.load('oksikiho_2014.model')
#学習済みのword2vecのモデルを読み込む
word_model = word2vec.Word2Vec.load('oksikiho_word2_2014.model')
#clf = svm.LinearSVC()

doc_list=[]
roa_list=[]
WORD_LIST_2014 = []
WORD_LIST_2015 = []

# 全てのファイルのリストを取得
def get_all_files(directory):
    #for文の変数が3つ
    for root, dirs, files in os.walk(directory):
        for file in files:
            #yieldはreturnのように処理終了せず一旦停止する
            #os.path.joinの返り値はroot
            #os.path.joinは1つあるいはそれ以上のパスの要素を賢く結合します。
            yield os.path.join(root, file)

def read_document(path):
#   with open(path, 'r', encoding='sjis', errors='ignore') as f:
    #with構文を使うと、close()の呼び出しが不要です。withブロックを抜けると、自動でclose()を呼び出してくれます。
    with open(path, 'r') as f:
        return f.read()            

# 青空文庫ファイルから作品部分のみ抜き出す
def trim_doc(doc):
    #splitlines():改行ごとに文字列を区切り, リストを返す
    lines = doc.splitlines()
    #空のリスト作成
    valid_lines = []
    is_valid = False
    horizontal_rule_cnt = 0
    break_cnt = 0
    for line in lines:
        if horizontal_rule_cnt < 2 and '-----' in line:
            horizontal_rule_cnt += 1
            is_valid = horizontal_rule_cnt == 2
            continue
        if not(is_valid):
            continue
        if line == '':
            break_cnt += 1
            is_valid = break_cnt != 3
            continue
        break_cnt = 0
        #要するに改行ごとに中身取り出して、作品部分だけを取り出す
        valid_lines.append(line)
        #''.join(valid_lines)で、''区切りで要素を連結している。
        #ここでは普通に連結しているだけ
    return ''.join(valid_lines)

# 文章から単語に分解して返す
def split_into_words(doc, name=''):
#    mecab = MeCab.Tagger("-Ochasen")
    #形態素解析
    mecab = MeCab("-Ochasen")
    #作品部分だけ抽出
    valid_doc = trim_doc(doc)
    #単語ごとに分割(linesはlist)
    lines = mecab.parse(doc).splitlines()
    words = []
    for line in lines:
        #水平タブごとに分割
        chunks = line.split('\t')
        if len(chunks) > 3 and (chunks[3].startswith('動詞') 
            or chunks[3].startswith('形容詞') 
            or (chunks[3].startswith('名詞') 
                and not chunks[3].startswith('名詞-数'))):
            #要は単語を抽出している
            WORD_LIST_2014.append(chunks[0])
            words.append(chunks[0])
    return LabeledSentence(words=words, tags=[name])

# ファイルから単語のリストを取得
def corpus_to_sentences(corpus):
    #ファイルをひとつずつ(x)読み込む
    #docsはlist型
    docs = [read_document(x) for x in corpus]
    #zip(a,b)でa,bを同時にループできる
    #enumerateでループ時にインデックス付きで要素を得ることができる
    #idx:インデックス、docs:本文のリスト、corpus:ファイル名のリスト
    for idx, (doc, name) in enumerate(zip(docs, corpus)):
        # 前処理中 今何番目か / 読み込む合計ファイル数 
        #sys.stdout.write('\r前処理中 {} / {} '.format(idx, len(corpus)))
        yield split_into_words(doc, name)

#回帰
def regression(corpus_doc,corpus_doc2):
	#global word_model
	# 2014年度の企業別のテキストとファイル名のペアのリストを作る
	sentences = list(corpus_to_sentences(corpus_doc))
	# 2015年度の企業別のテキストとファイル名のペアのリストを作る
	sentences2 = list(corpus_to_sentences(corpus_doc2))

	#各変数の初期化、構造定義
	X = np.empty((0,len(word_model[sentences[0].words[0]])), float)
	y = np.empty((0,1), float)
	test_X = np.empty((0,len(word_model[sentences2[0].words[0]])), float)
	test_y = np.empty((0,1), float)

	#全ファイル数分繰り返す
	for x in range(len(corpus_doc)):
		#print(sentences[x].tags)
		#sys.stdout.write('\r前処理中 {} / {} '.format(x, len(corpus_doc)))
		#ファイルの文章を取得
		docs = [read_document(i) for i in corpus_doc]
		#深層表現に対応するファイル名を抽出(実際には深層表現に一番近いファイル名を抽出
		#file_ = model.docvecs.most_similar([model.docvecs[x]])[0][0]
		file_ = sentences[x].tags[0]
		#ファイル名変換
		file_doc = file_.replace('./3roa-createOneInputFiledata/doc2014/', '')
		#対応するROAファイル名を作成
		file_roa = file_doc.replace('_2014txt', '_2015roa')
		#docファイルとROAファイルのリストを保存
		doc_list.append(file_doc)
		roa_list.append(file_roa)

		#対応するROAファイルの1行目を読み込む
		roa = linecache.getline('./3roa-createOneInputFiledata/roa2015/'+file_roa, 1)

		#各企業のベクトル表現を計算するための一時的なリスト
		tmp_array = np.empty((0,len(word_model[sentences[0].words[0]])), float) #初期化、構造定義
		#各企業の全単語について
		for i in range(len(sentences[x].words)):
			#学習済み単語(vocabに含まれている(min_count以上))以外のベクトルはエラーとなり
			#抽出できないためif文で学習単語以外をはじく
			if sentences[x].words[i] in word_model.wv.vocab:
				#単語ベクトルを格納
				tmp_array = np.append(X, np.array([word_model[sentences[x].words[i]]]), axis=0)
				#print(word_model.wv(sentences2[x].words[i]))

		#深層表現を格納
		#各企業の全単語ベクトルの平均を格納
		X = np.append(X, np.array([tmp_array.mean(axis = 0)]), axis=0)
		#ROAを格納
		y = np.append(y, np.array([[float(roa)]]), axis=0)

		#ROA読み込み時に使ったキャッシュを削除
		linecache.clearcache()

	#2015年度の文書から単語を抽出
	word_list2 = []
	for x in range(len(sentences2)):
		word_list2.append(sentences2[x].words)

	#既存のモデルのvocabを更新
	word_model.build_vocab(word_list2,update=True)
	#抽出した単語から学習
	word_model.train(word_list2,total_examples=word_model.corpus_count,epochs=word_model.iter)
	#word_model = train(sentences2)

	#予測データ	
	for j in range(100):
		#各企業のベクトル表現を計算するための一時的なリスト
		tmp_array = np.empty((0,len(word_model[sentences2[0].words[0]])), float)
		#各企業の全単語について
		for i in range(len(sentences2[j].words)):
			#学習済み単語(vocabに含まれている(min_count以上))以外のベクトルはエラーとなり
			#抽出できないためif文で学習単語以外をはじく
			if sentences2[j].words[i] in word_model.wv.vocab:
				#単語ベクトルを格納
				tmp_array = np.append(X, np.array([word_model[sentences2[j].words[i]]]), axis=0)

		#深層表現を格納
		#各企業の全単語ベクトルの平均を格納
		test_X = np.append(test_X, np.array([tmp_array.mean(axis = 0)]), axis=0)

		#ファイル名変換
		file_doc2 = corpus_doc2[j].replace('./3roa-createOneInputFiledata/doc2015/', '')
		#対応するROAファイル名を作成
		file_roa2 = file_doc2.replace('_2015txt', '_2016roa')

		#対応するROAファイルの1行目を読み込む
		roa2 = linecache.getline('./3roa-createOneInputFiledata/roa2016/'+file_roa2, 1)
		#ROAを格納
		test_y = np.append(test_y, np.array([[float(roa2)]]), axis=0)

	#ROA読み込み時に使ったキャッシュを削除
	linecache.clearcache()
	#2015年度のROA
	#avg = sum(y)/len(y) #標本の平均
	#print('y_average = %f' %(avg))	#標本の平均を出力
	#print('y_standard deviation = %f' %(np.std(y,dtype=float))) #標本標準偏差を出力

	#2016年度のROA
	#avg2 = sum(test_y)/len(test_y) #標本の平均
	#print('y_average = %f' %(avg2)) #標本の平均を出力
	#print('y_standard deviation = %f' %(np.std(test_y,dtype=float))) #標本標準偏差を出力

	#回帰
	#clf.fit(X, y)
	print('C=1e3,gamma=0.1,degree=3')
	#svr_rbf = SVR(kernel='rbf', C=1e3,gamma=0.1)
	#svr_lin = SVR(kernel='linear', C=1e3)
	#svr_poly = SVR(kernel='poly', C=1e3,degree=3)
	svr_rbf = SVR(kernel='rbf', C=1e3,gamma=0.1)
	svr_lin = SVR(kernel='linear', C=1e3)
	svr_poly = SVR(kernel='poly', C=1e3,degree=3)
	y_rbf = svr_rbf.fit(X, y).predict(X)
	y_lin = svr_lin.fit(X, y).predict(X)
	y_poly = svr_poly.fit(X, y).predict(X)

	print('y_rbf')
	for x in range(len(y_rbf)):
		print(y_rbf[x])

	print('y_lin')	
	for x in range(len(y_lin)):
		print(y_lin[x])

	print('y_poly')	
	for x in range(len(y_poly)):
		print(y_poly[x])
	

	print()

	#回帰の結果をもとに予測
	test_rbf = svr_rbf.predict(test_X)
	test_lin = svr_lin.predict(test_X)
	test_poly = svr_poly.predict(test_X)

	print('test_rbf')
	for x in range(len(test_rbf)):
		print(test_rbf[x])

	print('test_lin')	
	for x in range(len(test_lin)):
		print(test_lin[x])

	print('test_poly')	
	for x in range(len(test_poly)):
		print(test_poly[x])
	

	# 相関係数計算
	#rbf_corr = np.corrcoef(test_y, test_rbf)[0, 1]
	#lin_corr = np.corrcoef(test_y, test_lin)[0, 1]
	#poly_corr = np.corrcoef(test_y, test_poly)[0, 1]

	print()

	# RMSEを計算
	rbf_rmse = sqrt(mean_squared_error(test_y, test_rbf))
	lin_rmse = sqrt(mean_squared_error(test_y, test_lin))
	poly_rmse = sqrt(mean_squared_error(test_y, test_poly))

	#print "RBF: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr)
	#print "Linear: RMSE %f \t Corr %f" % (lin_rmse, lin_corr)
	#print "Poly: RMSE %f \t\t Corr %f" % (poly_rmse, poly_corr)

	# RMSEを出力
	print "RBF: RMSE %f " % (rbf_rmse)
	print "Linear: RMSE %f " % (lin_rmse)
	print "Poly: RMSE %f " % (poly_rmse)

	#print(corpus_doc2[1230])
	#print(corpus_doc2[2582])
	
# 学習
def train(sentences):
    # https://radimrehurek.com/gensim/models/doc2vec.html
    # size:特徴ベクトルの次元数,alpha:学習率の初期値(学習が進むにつれて学習率は線形的に0に低下する)
    # sample:任意の高頻度ワードがランダムにダウンサンプリングされるように構成するための閾値;デフォルトは0（オフ）、有効な値は1e-5です。
    # min_count:これよりも低い頻度のすべての単語を無視
    # workers:この数多くのワーカースレッドを使用してモデルをトレーニングします（=マルチコアマシンでのより速いトレーニング）。
    #model = models.Doc2Vec(size=1000, alpha=0.0015, sample=1e-4, min_count=3, workers=4)
    #model = models.Doc2Vec(size=400, alpha=0.0015, sample=1e-4, min_count=3, workers=4)
    #一連の文から語彙を構築する
    #model.build_vocab(sentences)
    #for x in range(30):
    for x in range(200):
        #現在のインデックスを表示
        print(x)
        model.train(sentences,total_examples=model.corpus_count,epochs=model.iter) #再学習するとここでセグメン
        #model.train(sentences,total_examples=model.corpus_count,epochs=100)
        ranks = []
        for doc_id in range(100):
            #infer_vector を使うとどうやら単語のリストをドキュメントベクトルに変換できるよっぽいらしい。
            #学習したモデルで新しい文書の内容を推測する
            inferred_vector = model.infer_vector(sentences[doc_id].words)
            # model.most_similar(positive=[単語]) で似ている単語が出せる
            # 独身女性 - 女性 + 男性 = ? model.most_similar(positive=[足す単語], negative=[引く単語])
            # model.most_similar(positive=['独身女性', '男性'], negative=['女性'])
            # model.docvecs.most_similar(positive=[◯◯])　これで文書の類似度出せる。加減算も可能
            # 各文書と最も類似度が高い文書を表示（デフォルト値：10個） 個数=topn？
            # sim = ('./3roa-createOneInputFiledata/8076.txt', 0.9981867671012878), ...
            sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
            # 引数で指定したオブジェクト(sentences[doc_id].tags[0])が持つ値が要素の中に含まれている場合は、
            # 最初の要素のインデックスを返します。
            rank = [docid for docid, sim in sims].index(sentences[doc_id].tags[0])
            # print(rank) インデックス
            # print(sentences[doc_id].tags[0]) ファイル名
            # print(sentences[doc_id].words) 単語名？
            # リストの最後に引数に指定したオブジェクト(rank)を追加します。
            ranks.append(rank)
            # 要素の数を数えあげて出力
        print(collections.Counter(ranks))
        if collections.Counter(ranks)[0] >= PASSING_PRECISION:
            break
    return model

if __name__ == '__main__':
    corpus_doc = list(get_all_files(INPUT_DOC_DIR))
    corpus_doc2 = list(get_all_files(INPUT_DOC_DIR2))
    regression(corpus_doc,corpus_doc2)
    print()

