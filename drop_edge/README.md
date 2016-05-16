# 実行方法

Pubchem上のデータを用いて実験することが想定されているため、pubchemから実験に用いるAssayデータ(.csv)とSubstanceデータ(.sdf)を用意しておく。

1.  data.py内のbase_path変数を設定する。
2.  python train.py [options]を実行する。オプションについてはall_masterと同様であるが、drop_masterには"-g,--gpu"オプションが付いておりCPUで実行するかGPUで実行するかを選ぶことができる。少なくとも現状ではGPUでの実行の方がtsubame上で２倍ほど遅い。

Tsubameにjobを投入する場合、stdoutをflushするか、python -u train.pyのように-uオプションをつけて実行するなどとしないと、実行が途中で強制終了した時に出力が得られなくなってしまうため注意が必要である。
