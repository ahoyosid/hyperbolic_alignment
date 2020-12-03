ENV_NAME=hypme
install:
	conda env create -f environment.yml &&\
	source ${HOME}/anaconda3/bin/activate $(ENV_NAME) &&\
	conda install pytorch torchvision -c pytorch &&\
    pip install geoopt &&\
	conda install -c conda-forge pot &&\
	conda deactivate

run_example:
	source ${HOME}/anaconda3/bin/activate $(ENV_NAME)  &&\
	python example.py &&\
	conda deactivate
