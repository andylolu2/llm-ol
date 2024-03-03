poetry-export-pip:
    poetry export --without-hashes --without-urls | awk '{ print $1 }' FS=';' > requirements.txt

cpu-run cmd cpus='8' mem='32G' time='6:00:00' *ARGS='':
    #!/bin/bash
    tmp_dir=$HOME/tmp/llm-ol-$(date +%s)
    mkdir -p $tmp_dir
    echo "Copying files to $tmp_dir"
    git ls-files --cached --others --exclude-standard | xargs -I {} cp --parents {} $tmp_dir
    ln -s /rds/user/cyal4/hpc-work/llm-ol/out $tmp_dir/out
    cd $tmp_dir
    sbatch --cpus-per-task={{cpus}} --mem={{mem}} --time={{time}} {{ARGS}} runs/launch.sh {{cmd}}

intr-cpu cpus='4' mem='10G' time='12:00:00':
    sbatch --cpus-per-task={{cpus}} --mem={{mem}} --time={{time}} runs/slurm/interactive_cpu.sh

clear-nb:
    find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;

code-count:
    cloc --vcs git
