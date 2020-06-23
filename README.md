XG-m7G
=======
XG-m7G is an computational approach based on XGBoost, which allows interested users to both use our model to identify N7-methylguanosine sites and train  their specific models based on their own datasets expediently.

* **--input:** RNA sequences to be predicted or self-training in fasta format. If you want to self train model, please put the label after the fasta name. For example: <br>
\>example1 1<br>
AAGAACAGGAGCGAGAGAAGGAGAGGGAAAAAGACAGAGAG<br>
\>example2 0<br>
CAGCGAGUUCGGUUGCGCGUGACGCACCGGGUGGGAGCGGA<br>

* **--purpose:** you can choose "predict" or "self-training" to indicate your purpose

* **--output:** Save the prediction  results in csv format
