```bash
./bin/cosfs2 --config-file=./conf/cosfs2.yaml cos://agentic-memory-dev-1310738255.cos-internal.ap-guangzhou.tencentcos.cn ./user-b/
```

```bash
time git commit -m "init commit"
[master (root-commit) a53af70] init commit
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 README.md

real    0m9.485s
user    0m0.005s
sys     0m0.012s


time git status
On branch master
nothing to commit, working tree clean

real    0m3.027s
user    0m0.002s
sys     0m0.006s

time git log
commit a53af70cb1de6c9a9dd813e7b3bb5251b8da9892 (HEAD -> master)
Author: limaoqiu <limaoqiu@tencent.com>
Date:   Wed May 6 20:44:58 2026 +0800

    init commit

real    0m2.594s
user    0m0.002s
sys     0m0.008s
```