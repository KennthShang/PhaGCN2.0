import  numpy as np
from    data import load_data, preprocess_features, preprocess_adj, sample_mask
import  model
from    config import  args
from    utils import masked_loss, masked_acc
import  pickle as pkl
import  scipy.sparse as sp
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F

import random

seed = 123
np.random.seed(seed)
torch.random.manual_seed(seed)


if torch.cuda.is_available():
    torch.cuda.set_device(0)
else:
    print("Running with cpu")


adj        = pkl.load(open(f"{args.outpath}/Cyber_data/contig.graph",'rb'))
labels     = pkl.load(open(f"{args.outpath}/Cyber_data/contig.label",'rb'))
features   = pkl.load(open(f"{args.outpath}/Cyber_data/contig.feature",'rb'))
test_to_id = pkl.load(open(f"{args.outpath}/Cyber_data/contig.dict",'rb'))
idx_test   = pkl.load(open(f"{args.outpath}/Cyber_data/contig.mask",'rb'))


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


pred_to_label = {199:"Belpaoviridae",103:"Bacilladnaviridae",137:"Closteroviridae",36:"Azeredovirinae",200:"Caulimoviridae",59:"Queuovirinae",127:"Pseudototiviridae",10:"Autographiviridae",194:"Potyviridae",112:"Nanoviridae",217:"Polydnaviriformidae",0:"Lipothrixviridae",38:"Bronfenbrennervirinae",126:"Fusagraviridae",33:"Zobellviridae",114:"Pecoviridae",179:"Fusariviridae",104:"Circoviridae",20:"Mesyanzhinovviridae",108:"Gandrviridae",115:"Naryaviridae",107:"Smacoviridae",2:"Alloherpesviridae",208:"Asfarviridae",31:"Vilmaviridae",125:"Chrysoviridae",66:"Alexandravirus",37:"Bclasvirinae",14:"Demerecviridae",170:"Peribunyaviridae",47:"Guarnerosvirinae",80:"Ignaciovirus",175:"Phenuiviridae",101:"Papillomaviridae",152:"Solspiviridae",207:"Iridoviridae",142:"Alphaflexiviridae",176:"Orthomyxoviridae",190:"Picornaviridae",145:"Flaviviridae",99:"Microviridae",164:"Nyamiviridae",72:"Casadabanvirus",7:"Hafunaviridae",13:"Chimalliviridae",62:"Stephanstirmvirinae",71:"Benedictvirus",94:"Veracruzvirus",118:"Kanorauviridae",193:"Solemoviridae",106:"Vilyaviridae",215:"Anelloviridae",18:"Herelleviridae",162:"Lispiviridae",148:"Mitoviridae",75:"Efquatrovirus",122:"Geplanaviridae",65:"Weiservirinae",149:"Atkinsviridae",166:"Rhabdoviridae",60:"Sepvirinae",189:"Marnaviridae",210:"Tectiviridae",213:"Nudiviridae",16:"Grimontviridae",214:"Nimaviridae",159:"Artoviridae",24:"Rountreeviridae",8:"Ackermannviridae",91:"Skunavirus",27:"Schitoviridae",30:"Umezonoviridae",48:"Guernseyvirinae",156:"Qinviridae",32:"Zierdtviridae",120:"Geminiviridae",51:"Joanripponvirinae",198:"Hepadnaviridae",163:"Mymonaviridae",82:"Kostyavirus",83:"Kroosvirus",181:"Partitiviridae",169:"Hantaviridae",12:"Chaseviridae",43:"Gclasvirinae",88:"Paclarkvirus",110:"Draupnirviridae",136:"Bromoviridae",143:"Betaflexiviridae",205:"Phycodnaviridae",76:"Fernvirus",63:"Tybeckvirinae",146:"Nodaviridae",109:"Ouroboviridae",64:"Vequintavirinae",105:"Endolinaviridae",42:"Eucampyvirinae",6:"Suoliviridae",81:"Korravirus",34:"Arquatrovirinae",78:"Gladiatorvirus",17:"Guelinviridae",121:"Genomoviridae",183:"Coronaviridae",61:"Skurskavirinae",128:"Spiciviridae",138:"Endornaviridae",165:"Paramyxoviridae",46:"Gracegardnervirinae",53:"Kutznervirinae",4:"Intestiviridae",11:"Casjensviridae",95:"Vividuovirus",206:"Mimiviridae",102:"Parvoviridae",124:"Botybirnaviridae",41:"Dolichocephalovirinae",23:"Pootjesviridae",157:"Aliusviridae",49:"Hendrixvirinae",70:"Beenievirus",178:"Curvulaviridae",25:"Saffermanviridae",216:"Fuselloviridae",67:"Andromedavirus",3:"Orthoherpesviridae",92:"Triavirus",58:"Pclasvirinae",86:"Mudcatvirus",184:"Mesoniviridae",39:"Ceeclamvirinae",26:"Salasmaviridae",151:"Fiersviridae",73:"Ceduovirus",150:"Duinviridae",158:"Chuviridae",173:"Arenaviridae",202:"Pseudoviridae",90:"Pbunavirus",15:"Drexlerviridae",9:"Aliceevansviridae",35:"Azeevirinae",56:"Nymbaxtervirinae",195:"Astroviridae",180:"Hypoviridae",123:"Pleolipoviridae",182:"Arteriviridae",129:"Artiviridae",55:"Nclasvirinae",68:"Attisvirus",119:"Mahapunaviridae",133:"Sedoreoviridae",161:"Filoviridae",211:"Adenoviridae",45:"Gorskivirinae",87:"Obolenskvirus",188:"Iflaviridae",187:"Dicistroviridae",79:"Gordonvirus",203:"Retroviridae",174:"Nairoviridae",201:"Metaviridae",116:"Adamaviridae",135:"Hepeviridae",28:"Stanwilliamsviridae",186:"Caliciviridae",212:"Baculoviridae",52:"Jondennisvirinae",168:"Fimoviridae",130:"Inseviridae",84:"Marthavirus",209:"Poxviridae",139:"Kitaviridae",155:"Botourmiaviridae",97:"Wizardvirus",160:"Bornaviridae",96:"Wbetavirus",153:"Blumeviridae",167:"Xinmoviridae",134:"Spinareoviridae",111:"Anicreviridae",132:"Orthototiviridae",5:"Steigviridae",113:"Redondoviridae",144:"Tymoviridae",191:"Polycipiviridae",141:"Virgaviridae",131:"Lebotiviridae",89:"Pahexavirus",85:"Montyvirus",172:"Tospoviridae",69:"Backyardiganvirus",54:"Mccleskeyvirinae",192:"Secoviridae",100:"Polyomaviridae",98:"Inoviridae",19:"Kyanoviridae",74:"Dhillonvirus",197:"Birnaviridae",50:"Jameshumphriesvirinae",21:"Orlajensenviridae",147:"Tombusviridae",1:"Rudiviridae",204:"Polymycoviridae",196:"Yadokariviridae",177:"Amalgaviridae",154:"Steitzviridae",140:"Togaviridae",29:"Straboviridae",44:"Gordonclarkvirinae",57:"Ounavirinae",117:"Kirkoviridae",171:"Phasmaviridae",22:"Peduoviridae",185:"Tobaniviridae",93:"Turbidovirus",40:"Deejayvirinae",77:"Fromanvirus"}

with open(f"{args.outpath}/prediction.csv", 'w') as f_out:
    _ = f_out.write("contig_names, prediction\n")
    for key in test_to_id.keys():
        if labels[test_to_id[key]] == -1:
            _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]]) + "\n")
        elif labels[test_to_id[key]] == -2:
            _ = f_out.write(str(key) + "," + str(pred_to_label[pred[test_to_id[key]]])+"_like" + "\n")
        else:
            _ = f_out.write(str(key) + "," + str(pred_to_label[labels[test_to_id[key]]]) + "\n")


