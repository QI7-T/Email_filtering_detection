srun -p A40 -N 1 -n 2 --gres=gpu:1 --exclude=comput[01] --pty /bin/bash
