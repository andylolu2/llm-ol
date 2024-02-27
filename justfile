poetry-export-pip:
    poetry export --without-hashes --without-urls | awk '{ print $1 }' FS=';' > requirements.txt

cpu-run cmd cpus=8 mem=32G time=6:00:00 *ARGS:
    #!/bin/bash
    tmp_dir=/tmp/llm-ol-$(date +%s)
    git clone . $tmp_dir
    cd $tmp_dir
    ln -s /rds/user/cyal4/hpc-work/llm-ol/out out
    sbatch \
        --cpus-per-task={{cpus}} \
        --mem={{mem}} \
        --time={{time}} \
        {{ARGS}} \
        runs/launch.sh {{cmd}}

clear-nb:
    find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

code-count:
    cloc --vcs git