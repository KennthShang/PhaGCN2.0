import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
import  numpy as np
from    data import load_data, preprocess_features, preprocess_adj, sample_mask
import  model
from    config import  args
from    utils import masked_loss, masked_acc
import  pickle as pkl
import  scipy.sparse as sp
from sklearn.metrics import classification_report

import random

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    print("Running with cpu")


adj        = pkl.load(open("Cyber_data/contig.graph",'rb'))
labels     = pkl.load(open("Cyber_data/contig.label",'rb'))
features   = pkl.load(open("Cyber_data/contig.feature",'rb'))
test_to_id = pkl.load(open("Cyber_data/contig.dict",'rb'))
idx_test   = pkl.load(open("Cyber_data/contig.mask",'rb'))

idx_test = np.array(idx_test)
labels = np.array(labels)

y_train = np.zeros(labels.shape)
y_test = np.zeros(labels.shape)



idx_train = np.array([i for i in range(len(labels)) if i not in idx_test])


train_mask = sample_mask(idx_train, labels.shape[0])
test_mask = sample_mask(idx_test, labels.shape[0])

y_train[train_mask] = labels[train_mask]
y_test[test_mask] = labels[test_mask]


features = sp.csc_matrix(features)

print('adj:', adj.shape)
print('features:', features.shape)
print('y:', y_train.shape, y_test.shape) # y_val.shape, 
print('mask:', train_mask.shape, test_mask.shape) # val_mask.shape

features = preprocess_features(features) # [49216, 2], [49216], [2708, 1433]
supports = preprocess_adj(adj)

if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device('cuda')
    train_label = torch.from_numpy(y_train).long().to(device)
    num_classes = max(labels)+1
    train_mask = torch.from_numpy(train_mask.astype(np.bool)).to(device)
    test_label = torch.from_numpy(y_test).long().to(device)
    test_mask = torch.from_numpy(test_mask.astype(np.bool)).to(device)

    i = torch.from_numpy(features[0]).long().to(device)
    v = torch.from_numpy(features[1]).to(device)
    feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float().to(device)

    i = torch.from_numpy(supports[0]).long().to(device)
    v = torch.from_numpy(supports[1]).to(device)
    support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)

else:
    train_label = torch.from_numpy(y_train).long()
    num_classes = max(labels)+1
    train_mask = torch.from_numpy(train_mask.astype(np.bool))
    test_label = torch.from_numpy(y_test).long()
    test_mask = torch.from_numpy(test_mask.astype(np.bool))

    i = torch.from_numpy(features[0]).long()
    v = torch.from_numpy(features[1])
    feature = torch.sparse.FloatTensor(i.t(), v, features[2]).float()

    i = torch.from_numpy(supports[0]).long()
    v = torch.from_numpy(supports[1])
    support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float()

print('x :', feature)
print('sp:', support)
num_features_nonzero = feature._nnz()
feat_dim = feature.shape[1]


def accuracy(out, mask):
    pred = np.argmax(out, axis = 1)
    mask_pred = np.array([pred[i] for i in range(len(labels)) if mask[i] == True])
    mask_label = np.array([labels[i] for i in range(len(labels)) if mask[i] == True])
    return np.sum(mask_label == mask_pred)/len(mask_pred)

net = model.GCN(feat_dim, num_classes, num_features_nonzero)
if torch.cuda.is_available():
    net.to(device)
optimizer = optim.Adam(net.parameters(), lr=0.01)#args.learning_rate


_ = net.train()
for epoch in range(args.epochs*2):
    # forward pass
    out = net((feature, support))
    #out = out[0]
    loss = masked_loss(out, train_label, train_mask)
    loss += args.weight_decay * net.l2_loss()
    # backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # output
    if epoch % 10 == 0:
        # calculating the acc
        _ = net.eval()
        out = net((feature, support))
        if torch.cuda.is_available():
            acc_train = accuracy(out.detach().cpu().numpy(), train_mask.detach().cpu().numpy())
        else:
            acc_train = accuracy(out.detach().numpy(), train_mask.detach().numpy())
        #acc_test = accuracy(out.detach().cpu().numpy(), test_mask.detach().cpu().numpy())
        print(epoch, loss.item(), acc_train)
        if acc_train > 0.978:
            break
    _ = net.train()


net.eval()
out = net((feature, support))
if torch.cuda.is_available():
    out = out.cpu().detach().numpy()
else:
    out = out.detach().numpy()

pred = np.argmax(out, axis = 1)

mode = "testing"
if mode == "validation":
    print(classification_report(labels, pred))
    print(accuracy(out, train_mask.detach().cpu().numpy()))
    mask = test_mask.detach().cpu().numpy()
    test_pred = np.array([pred[i] for i in range(len(pred)) if mask[i] == True])
    test_label = np.array([labels[i] for i in range(len(labels)) if mask[i] == True])
    print(classification_report(test_label, test_pred))
    print(np.sum(test_label == test_pred)/len(test_pred))


pred_to_label = {50:"Andromedavirus",99:"Alphaflexiviridae",105:"Mitoviridae",167:"Baculoviridae",141:"Mesoniviridae",19:"Peduoviridae",139:"Arteriviridae",118:"Lispiviridae",54:"Ceduovirus",44:"Queuovirinae",13:"Drexlerviridae",113:"Aliusviridae",26:"Zobellviridae",56:"Fernvirus",165:"Tectiviridae",142:"Tobaniviridae",34:"Dolichocephalovirinae",0:"Lipothrixviridae",70:"Turbidovirus",65:"Pahexavirus",159:"Retroviridae",48:"Vequintavirinae",140:"Coronaviridae",110:"Steitzviridae",38:"Guernseyvirinae",78:"Parvoviridae",136:"Fusariviridae",137:"Hypoviridae",157:"Metaviridae",6:"Suoliviridae",79:"Bacilladnaviridae",40:"Mccleskeyvirinae",60:"Kostyavirus",64:"Obolenskvirus",145:"Iflaviridae",106:"Atkinsviridae",88:"Chrysoviridae",148:"Polycipiviridae",135:"Curvulaviridae",152:"Astroviridae",41:"Nclasvirinae",160:"Polymycoviridae",2:"Alloherpesviridae",144:"Dicistroviridae",42:"Ounavirinae",52:"Benedictvirus",67:"Pbunavirus",156:"Caulimoviridae",73:"Wizardvirus",33:"Deejayvirinae",149:"Secoviridae",107:"Fiersviridae",151:"Potyviridae",153:"Birnaviridae",9:"Autographiviridae",17:"Mesyanzhinovviridae",43:"Pclasvirinae",77:"Papillomaviridae",49:"Weiservirinae",12:"Demerecviridae",84:"Redondoviridae",129:"Peribunyaviridae",4:"Intestiviridae",27:"Arquatrovirinae",24:"Vilmaviridae",39:"Hendrixvirinae",51:"Backyardiganvirus",85:"Geminiviridae",111:"Botourmiaviridae",14:"Guelinviridae",121:"Paramyxoviridae",76:"Polyomaviridae",164:"Poxviridae",86:"Genomoviridae",3:"Herpesviridae",104:"Tombusviridae",97:"Togaviridae",89:"Totiviridae",94:"Closteroviridae",127:"Hantaviridae",59:"Korravirus",125:"Arenaviridae",36:"Gclasvirinae",15:"Herelleviridae",10:"Casjensviridae",101:"Tymoviridae",133:"Orthomyxoviridae",32:"Ceeclamvirinae",82:"Smacoviridae",30:"Boydwoodruffvirinae",80:"Circoviridae",119:"Mymonaviridae",8:"Ackermannviridae",128:"Nairoviridae",130:"Phasmaviridae",25:"Zierdtviridae",81:"Vilyaviridae",66:"Pakpunavirus",92:"Hepeviridae",117:"Filoviridae",31:"Bronfenbrennervirinae",95:"Endornaviridae",126:"Fimoviridae",132:"Tospoviridae",68:"Skunavirus",75:"Microviridae",71:"Veracruzvirus",98:"Virgaviridae",83:"Nanoviridae",5:"Steigviridae",63:"Montyvirus",21:"Salasmaviridae",1:"Rudiviridae",74:"Inoviridae",109:"Blumeviridae",134:"Amalgaviridae",57:"Fromanvirus",87:"Pleolipoviridae",28:"Azeredovirinae",154:"Hepadnaviridae",120:"Nyamiviridae",170:"Anelloviridae",18:"Orlajensenviridae",16:"Kyanoviridae",47:"Tybeckvirinae",171:"Fuselloviridae",45:"Sepvirinae",162:"Iridoviridae",55:"Efquatrovirus",158:"Pseudoviridae",96:"Kitaviridae",102:"Flaviviridae",163:"Asfarviridae",161:"Phycodnaviridae",143:"Caliciviridae",7:"Hafunaviridae",150:"Solemoviridae",112:"Qinviridae",123:"Rhabdoviridae",58:"Gladiatorvirus",22:"Schitoviridae",116:"Bornaviridae",72:"Vividuovirus",108:"Solspiviridae",131:"Phenuiviridae",103:"Nodaviridae",90:"Sedoreoviridae",46:"Stephanstirmvirinae",146:"Marnaviridae",53:"Casadabanvirus",114:"Chuviridae",62:"Marthavirus",11:"Chaseviridae",168:"Nudiviridae",166:"Adenoviridae",100:"Betaflexiviridae",115:"Artoviridae",169:"Nimaviridae",124:"Xinmoviridae",61:"Kroosvirus",122:"Pneumoviridae",69:"Triavirus",91:"Spinareoviridae",93:"Bromoviridae",138:"Partitiviridae",23:"Straboviridae",35:"Eucampyvirinae",147:"Picornaviridae",155:"Belpaoviridae",20:"Rountreeviridae",37:"Gracegardnervirinae",29:"Bclasvirinae"}

with open("prediction.csv", 'w') as f_out:
    _ = f_out.write("contig_names, prediction\n")
    for key in test_to_id.keys():
        if labels[test_to_id[key]] == -1:
            _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]]) + "\n")
        elif labels[test_to_id[key]] == -2:
            _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]])+"_like" + "\n")
        else:
            _ = f_out.write(str(key) + "," + str(pred_to_label[labels[test_to_id[key]]]) + "\n")


