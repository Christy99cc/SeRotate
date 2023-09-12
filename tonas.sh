#!/bin/bash

# ./toserver.sh room@15123  exp_name
# ./toserver.sh room@15123 phasen_torch se_phasen_009

if [ -z "$1" ]; then
  echo "Need a destination."
  exit 0
fi
#site=${1#*@}
#user=${1%@*}
site=$1
#rm _data _log -rf
#rm *__pycache__* -rf
#rm */__pycache__* -rf


if [ "$site" == "80" ]; then
  rsync -rltDvz -e ./../* /share/program/zsx2/projects/mmrotate80/

elif [ "$site" == "40" ]; then
  rsync -rltDvz -e ./../* /share/program/zsx2/projects/mmrotate40/
fi
# -a ：递归到目录，即复制所有文件和子目录。另外，打开归档模式和所有其他选项（相当于 -rlptgoD）
# -v ：详细输出
# -e ssh ：使用 ssh 作为远程 shell，这样所有的东西都被加密
# --exclude='*.out' ：排除匹配模式的文件，例如 *.out 或 *.c 等。

