import ipfshttpclient

import dataset
import parsers

test_hash = 'bafybeic7qbuo2ail2y5urbm5btfp7dwcxigjs4kq6m36ecbozaurt4z3te'

client = ipfshttpclient.connect()

data = dataset.IPFSGeneralDataset(client, 'data', [test_hash], parser=parsers.IPFSImageTensorParser())
print(data[0].shape)

#dataset.IPFSGeneralDataset(client, '', [test_hash])