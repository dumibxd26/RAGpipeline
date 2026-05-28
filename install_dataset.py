from beir import util
from beir.datasets.data_loader import GenericDataLoader
# Template-ul URL pentru arhivele BEIR găzduite de UKP Darmstadt
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
# Descarcă arhiva setului de date și o dezarhivează în folderul "datasets"
# (înlocuiți "scifact" cu "nfcorpus" sau "fiqa" pentru celelalte seturi)
data_path = util.download_and_unzip(url.format("fiqa"), "datasets")
# Încarcă cele trei structuri necesare:
# corpus — dicționar {doc_id: {"title": ..., "text": ...}}
# queries — dicționar {query_id: text_query}
# qrels — judecățile de relevanță {query_id: {doc_id: scor_relevanță}}
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")