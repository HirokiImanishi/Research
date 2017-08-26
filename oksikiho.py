#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# http://qiita.com/akira_/items/f9bb46cad6834da32367
# https://github.com/Foo-x/doc2vec-sample

import codecs
import os
import sys
from natto import MeCab
import collections
import numpy as np
from gensim import models
from gensim.models.doc2vec import LabeledSentence
from gensim.models import word2vec

#それぞれ2014,15,16年度の四季報のテキストのフォルダ
INPUT_DOC_DIR = './3roa-createOneInputFiledata/doc2014'
INPUT_DOC_DIR2 = './3roa-createOneInputFiledata/doc2015'
INPUT_DOC_DIR3 = './3roa-createOneInputFiledata/doc2016'
#doc2vecの出力モデルの名前
OUTPUT_MODEL = 'oksikiho_2014.model'
#word2vecの出力モデルの名前
OUTPUT_MODEL2 = 'oksikiho_word2_2014.model'
PASSING_PRECISION = 93

# 全てのファイルのリストを取得
def get_all_files(directory):
    #for文の変数が3つ
    for root, dirs, files in os.walk(directory):
        for file in files:
            #yieldはreturnのように処理終了せず一旦停止する
            #os.path.joinの返り値はroot
            #os.path.joinは1つあるいはそれ以上のパスの要素を賢く結合します。
            yield os.path.join(root, file)

# ファイルから文章を返す
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
            #WORD_LIST.append(chunks[0])
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
        sys.stdout.write('\r前処理中 {} / {} '.format(idx, len(corpus)))
        yield split_into_words(doc, name)

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
        model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
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
    #list型はC言語でいう配列
    #corpusには読み込んだ全ファイル名が入っている
    #global TRAIN_COUNT

    # 2014年度の全てのファイルのリストを取得
    corpus = list(get_all_files(INPUT_DOC_DIR))
    # 2014年度の企業別のテキストとファイル名のペアのリストを作る
    sentences = list(corpus_to_sentences(corpus))
    print("corpus OK")

    # 2015年度の全てのファイルのリストを取得
    corpus2 = list(get_all_files(INPUT_DOC_DIR2))
    # 2015年度の企業別のテキストとファイル名のペアのリストを作る
    sentences2 = list(corpus_to_sentences(corpus2))
    print("corpus2 OK")

    #doc2vecのモデルを作る
    #model = models.Doc2Vec(sentences,size=400, alpha=0.0015, sample=1e-4, min_count=3, workers=4)
    model = models.Doc2Vec(size=400, alpha=0.0015, sample=1e-4, min_count=3, workers=4)
    #語彙をビルドする
    model.build_vocab(sentences)
    #学習
    #model.train(sentences,total_examples=model.corpus_count,epochs=model.iter)
    model = train(sentences)
    print("doc_sentences OK")
    #学習したdoc2vecのモデルを保存
    model.save(OUTPUT_MODEL)

    #word2vecのモデルを作成
    #2014年度の文書から単語を抽出
    word_list = [] 
    for x in range(len(sentences)):
        word_list.append(sentences[x].words)
    #抽出した単語から学習
    word_model = word2vec.Word2Vec(word_list)
    word_model.train(word_list,total_examples=word_model.corpus_count,epochs=word_model.iter)
    #word_model = train(sentences)
    print("word_sentences OK")
    #学習したword2vecのモデルを保存
    word_model.save(OUTPUT_MODEL2)
    print()
    #print(str(sentences[0].words).decode("string-escape")) #printでリストの文字化けを防ぐ方法
