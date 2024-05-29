# GraphRA
Implementing the RAG(RA) technique to facilitate the LLMs in graph information learning, achieving 73.29% on the obgn-arxiv dataset.<br>
This project is used as the final research project of COMP4880 Computational Methods for Network Scinece at The Australian National University (ANU).<br>
The contributors are 1th. Oucheng Liu and 2th. Zezhou Wang, from ANU.<br>


# To run the GraphRA project, install the necessary library in following sequence (we assume torch version 2.1.0 and CUDA 12.1): <br>

```
pip install bitsandbytes
```
```
pip install git+https://github.com/huggingface/transformers.git
```
```
pip install git+https://github.com/huggingface/peft.git
```
```
pip install git+https://github.com/huggingface/accelerate.git
```
```
pip install datasets
```
```
pip install trl
```
```
pip install torch_geometric
```
```
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
```
```
pip install ogb
```
# To facilitate operations, we have processed the embeddings and top-k neighbor nodes that will be used during training and placed them in the data folder

