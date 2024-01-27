poetry-export-pip:
    poetry export --without-hashes --without-urls | awk '{ print $1 }' FS=';' > requirements.txt

srun *ARGS:
    sbatch runs/launch.sh {{ARGS}}

clear-nb:
    find . -name "*.ipynb" -exec jupyter nbconvert --clear-output --inplace {} \;