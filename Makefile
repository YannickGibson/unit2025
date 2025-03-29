vis: 
	cd visualization && python -m streamlit run vis.py --server.headless true | cat
train:
	cd model && python model.py