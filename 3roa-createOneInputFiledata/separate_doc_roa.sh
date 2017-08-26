#!/bin/bash
IFS_BACKUP=$IFS
IFS=$'\n'
#document部分のみを抽出した.txtファイルを年度ごとに保存するフォルダを作成
mkdir -m +w doc2014
mkdir -m +w doc2015
mkdir -m +w doc2016
#ROA部分のみを抽出した.txtファイルを年度ごとに保存するフォルダを作成
mkdir -m +w roa2014
mkdir -m +w roa2015
mkdir -m +w roa2016

#フォルダ内の.txtがつくファイルを全て読み込む
for file in $(ls *.txt)
do
    #行数カウンタ(今何行目を読み込んでいるか)
    count=1
    #読み込んだファイルを1行ずつ読み込む
    for line in `cat ${file}`
    do

    #3行目を読み込む時(2014年度)
    if [ $count == 3 ]; then
        #ROAのみを抽出して新たなファイルに保存
        roa=`echo $line | sed -e 's/<ROA:\(.*\)>.*$/\1/'`
        str="_2014roa.txt"
        tmp=`echo $file | sed -e "s/\.txt//"`
        newfile=$tmp${str}
        echo ${roa} >>roa2014/$newfile

        #本文のみを抽出して新たなファイルに保存
        doc=`echo $line | sed -e 's/^.*>//'`
        str="_2014txt.txt"
        tmp=`echo $file | sed -e "s/\.txt//"`
        newfile=$tmp${str}
        echo ${doc} >>doc2014/$newfile
    fi

    #4行目を読み込む時(2015年度)
    if [ $count == 4 ]; then
        #ROAのみを抽出して新たなファイルに保存
        roa=`echo $line | sed -e 's/<ROA:\(.*\)>.*$/\1/'`
        str="_2015roa.txt"
        tmp=`echo $file | sed -e "s/\.txt//"`
        newfile=$tmp${str}
        echo ${roa} >>roa2015/$newfile

        #本文のみを抽出して新たなファイルに保存
        doc=`echo $line | sed -e 's/^.*>//'`
        str="_2015txt.txt"
        tmp=`echo $file | sed -e "s/\.txt//"`
        newfile=$tmp${str}
        echo ${doc} >>doc2015/$newfile
    fi

    #5行目を読み込む時(2016年度)
    if [ $count == 5 ]; then
        #ROAのみを抽出して新たなファイルに保存
        roa=`echo $line | sed -e 's/<ROA:\(.*\)>.*$/\1/'`
        str="_2016roa.txt"
        tmp=`echo $file | sed -e "s/\.txt//"`
        newfile=$tmp${str}
        echo ${roa} >>roa2016/$newfile

        #本文のみを抽出して新たなファイルに保存
        doc=`echo $line | sed -e 's/^.*>//'`
        str="_2016txt.txt"
        tmp=`echo $file | sed -e "s/\.txt//"`
        newfile=$tmp${str}
        echo ${doc} >>doc2016/$newfile
    fi
    count=$(($count+1)) #次の行数に合わす為のインクリメント
    done
done
IFS=$IFS_BACKUP
