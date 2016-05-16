# 実行方法

Pubchem上のデータを用いて実験することが想定されているため、pubchemから実験に用いるAssayデータ(.csv)とSubstanceデータ(.sdf)を用意しておく。

1.  data.py内のbase_path変数（データの置かれているパス）を設定する。
2.  python train.py [options]を実行する。-hオプションをつけて実行するとオプションの詳細が表示される。特に、"-a,--assay"オプションを指定することでpubchem上でのAssayIDを指定することができる。


Tsubameにjobを投入する場合、stdoutをflushするか、python -u train.pyのように-uオプションをつけて実行するなどとしないと、実行が途中で強制終了した時に出力が得られなくなってしまうため注意が必要である。
