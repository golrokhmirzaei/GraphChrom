# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 14:49:35 2022

@author: mirzaei.4
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import os.path as osp
from matplotlib import pyplot as plt
from torch_geometric.data import Data, DataLoader,Dataset
import sklearn 
import math
import copy
import itertools
from torch_geometric.utils import to_networkx, from_networkx,to_dense_adj
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import WeightedRandomSampler
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import sklearn.exceptions


classes= {1:'breast', 2:'pancreas', 3:'prostate'} 
num_epochs=200
num_classes=3
hidden_dim=40
training_rate=0.8
num_node_features = 24
numOfChrom = 24
avg_con_threshold=0.3
batch_size=10



class Chrom():   
    def __init__(self):
        self.node_feature=[]
        self.edge_connection_x=[]
        self.edge_connection_y=[]
        self.label=""
        self.labelCode=-1
        self.nodes=[]
        self.index=0
        self.loaded=False
        self.location_from=[]                     
        self.location_to=[]                        
       # self.node_indegree=0
       # self.node_outdegree=0
        
    def setIndex(self, index):
        self.index=index
        self.loaded=True
        
    def setLabel(self, label):
        self.label=label
        
    def setLabelCode(self, labelCode):
        self.labelCode=labelCode
        
    def setNodes(self):
        for i in range(numOfChrom):
            self.nodes.append(i+1)
            
    def addEdgeConnection(self, x, y):
        self.edge_connection_x.append(x)
        self.edge_connection_y.append(y)
                
    def resetAttrib(self):
        self.node_feature=[]
        self.edge_connection_x=[]
        self.edge_connection_y=[]
        self.nodes=[]
        return
    
    
class MyOwnDataset(Dataset):
    def __init__(self, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(None, transform, pre_transform)


def convertLabelToInt(label,labels):     # labe is string and labels is a list
    for i in range(len(labels)):
        if labels[i][0]==label:
            labelCode=labels[i][1]-1
            return int(labelCode)
    return -100


def matrix_to_list(matrix):
    graph = {}
    for i, node in enumerate(matrix):
        adj = []
        for j, connected in enumerate(node):
            if connected:
                adj.append(j)
        graph[i] = adj
    return graph


def find_nodes_dic(glist):
    for k, v in glist.items():
        if v!='':
            return k, v

def bfs(graph, node): 
  queue=[]
  visited=[]
  visited.append(node)
  queue.append(node)

  while queue:          
    m = queue.pop(0) 
    #print (m, end = " ") 

    for neighbour in graph[m]:
      if neighbour not in visited:
        visited.append(neighbour)
        queue.append(neighbour)
  return visited 
    
def convertToAdjMatrix(edgeIndex):    
   adj_mtrx=torch.tensor(24) 
   adj_mtrx=to_dense_adj(g.edge_index,max_num_nodes=24).numpy()
   return adj_mtrx   

def convertToAdjList(edgeIndex):    
   adj_mtrx=torch.tensor(24) 
   adj_mtrx=to_dense_adj(g.edge_index,max_num_nodes=24).numpy()
   adj_list=matrix_to_list(adj_mtrx.reshape(24,24))
   return adj_list   


from collections import defaultdict

#This class represents a directed graph using
# adjacency matrix representation
class Graph:

    def __init__(self,graph):
        self.graph = graph
        self. ROW = len(graph)

    '''Returns true if there is a path from source 's' to sink 't' in
    residual graph. Also fills parent[] to store the path '''
    def BFS(self,s, t, parent):

        # Mark all the vertices as not visited
        visited =[False]*(self.ROW)

        # Create a queue for BFS
        queue=[]

        # Mark the source node as visited and enqueue it
        queue.append(s)
        visited[s] = True

        # Standard BFS Loop
        while queue:

            #Dequeue a vertex from queue and print it
            u = queue.pop(0)

            # Get all adjacent vertices of the dequeued vertex u
            # If a adjacent has not been visited, then mark it
            # visited and enqueue it
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0 :
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u

        # If we reached sink in BFS starting from source, then return
        # true, else false
        return True if visited[t] else False



    # Returns tne maximum number of edge-disjoint paths from
    #s to t in the given graph
    def findDisjointPaths(self, source, sink):

        # This array is filled by BFS and to store path
        parent = [-1]*(self.ROW)

        max_flow = 0 # There is no flow initially

        # Augment the flow while there is path from source to sink
        while self.BFS(source, sink, parent) :

            # Find minimum residual capacity of the edges along the
            # path filled by BFS. Or we can say find the maximum flow
            # through the path found.
            path_flow = float("Inf")
            s = sink
            while(s != source):
                path_flow = min (path_flow, self.graph[parent[s]][s])
                s = parent[s]

            # Add path flow to overall flow
            max_flow += path_flow

            # update residual capacities of the edges and reverse edges
            # along the path
            v = sink
            while(v != source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]

        return max_flow

           
# =============================================================================
# # Reading Data    
# =============================================================================
root = 'D:/research- tenure track/graph datasets/Aberations/'
df_data = pd.read_excel('D:/data.xlsx',engine="openpyxl")
data = df_data.values.tolist()        
df_labels = pd.read_excel('D:/Labels_3way.xlsx', engine="openpyxl")
labels = df_labels.values.tolist()       
#create empty list of graphs
list=[]                     

# =============================================================================
# # Create graphs and store in "list"
# =============================================================================
for sample in range(len(data)):
    
    if (sample == 0):
        
        #read data related to the first graph
        graph_index = 0                       #index 0 for the first graph
        current_sample_name = data[sample][0] #sample name starts with first graph name and will be update

        #temporary object to the first graph
        temp = Chrom()

        #initilaze the first graph information including the first connection
        temp.setIndex(graph_index)       
        temp.setLabel(data[sample][4])       
        labelCode=convertLabelToInt(data[sample][4],labels)
        temp.setLabelCode(labelCode)
        temp.setNodes()
        temp.addEdgeConnection(data[sample][2]-1, data[sample][3]-1)
        
        #if this is the only connection add it to list of graphs
        if (sample == len(data)-1):  # or leng(data) == 1
            list.append(temp)

    else: # sample is not the first element

        #adding connection to current graph    
        if (current_sample_name == data[sample][0]):        
            temp.addEdgeConnection(data[sample][2]-1, data[sample][3]-1)
        
            #add last graph to list of graphs
            if (sample == len(data)-1):
                list.append(temp)
        else:    
            if (temp.loaded):
                list.append(temp)        
            #load new graph starting with the first connection
            temp = Chrom() 
        
            current_sample_name = data[sample][0]
            graph_index = graph_index + 1
            temp.setIndex(graph_index)
            labelCode=convertLabelToInt(data[sample][4],labels)
            temp.setLabelCode(labelCode)
            temp.setLabel(data[sample][4]) 
            temp.setNodes()
            temp.addEdgeConnection(data[sample][2]-1, data[sample][3]-1)
# =============================================================================
# # create edges ( with no duplication) and edge attributes : if there is for example 2 same edges, we set the attribute of grapg as 2
# =============================================================================
data_list =[]
for i in range(len(list)):
    temp=list[i]
    edge_connection_x=temp.edge_connection_x
    edge_connection_y=temp.edge_connection_y
    edge_connection_xy=[edge_connection_x, edge_connection_y]
    edge_connection_xy=np.array(edge_connection_xy)
    # count frequency of each edge    
    unique_rows = np.unique(edge_connection_xy, axis=1)               
    edge_attr=[]
    for i in range(len (unique_rows[0])):
        count=0
        for j in range(len(edge_connection_xy[0])):
            a=unique_rows[:,i]
            b=edge_connection_xy[:,j]
            if a[0]==b[0] and a[1]==b[1]:
                count=count+1
        edge_attr.append(count)
    edge_index = torch.tensor(unique_rows, dtype=torch.long)
    y= temp.labelCode       # y is the target
    # features are considered as identity matrix
    node_features= torch.eye(24)         
    node_features=np.array(node_features)    
    x = torch.from_numpy(node_features).type(torch.float32)
    # create objects of type Data which are the pytorch gyometric graphs
    G = Data(x=x,edge_index=edge_index.to(torch.long), edge_attr =edge_attr, y=y) 
    data_list.append(G)
    
#Count the number of targets in the whole data set
sum0=0
sum1=0
sum2=0

for i in range (len(data_list)):
    if data_list[i].y==0:              
        sum0=sum0+1
    elif data_list[i].y==1:
        sum1=sum1+1
    elif data_list[i].y==2:
        sum2=sum2+1

print ("sum0:" , sum0) 
print ("sum1:" , sum1)   
print ("sum2:" , sum2) 

# =============================================================================
# Part II :  Create  data_list_append which is list of graphs containing following attributes:x, edge_index, edge_attr, y, reachable_nodes, weight
# =============================================================================
print('Part II: calculate graph appendix: average graph connectivity, number of nodes, the name of nodes (unique), weight, reachable nodes from each node')
del list    # to avoid the error in the next line: "list object is not callable"
# for each graph we need the total weights , list of reachable nodes, list of satisfied nodes(that met a specific criteria)

class DataAppend():
    def __init__(self, x, edge_index, edge_attr, y):

        self.reachable_nodes=[]
        self.weight=[]
        self.edge_index_list=[]
        self.max_disjoint_path=np.zeros((24,24))
        self.avg_con=0
        self.edge_index=edge_index
        self.edge_attr=edge_attr
        self.x=x
        self.y=y
        self .num_nodes=0
        self.unique_nodes=[]

        
    def setReachableNodes(self, reachable_nodes):
        self.reachable_nodes=reachable_nodes
        
    def setWeight(self, weight):
        self.weight=weight
        
    def setMaxDisjointPath(self,max_disjoint_path):
        self.max_disjoint_path=max_disjoint_path
        
    def setAvgCon(self,avg_con):
        self.avg_con=avg_con
    
    def setNumNodes(self,num_nodes):
        self.num_nodes=num_nodes
        
    def setUniqueNodes(self,unique_nodes):
        self.unique_nodes=unique_nodes
        

data_list_append=[]        

for i in range (len(data_list)):
    
    g=data_list[i]
    temp=DataAppend(g.x,g.edge_index,g.edge_attr, g.y)
    # convert edge_index to adjancy matrix and then to adjancy list as the appropriate input for BFS
    edge_index_list=convertToAdjList(g.edge_index)
    
    # calculate reachable nodes in adjancy list using BFS (Breadth First Seartch)
    g_reachable_nodes=[]
    for j in range (numOfChrom-1):
        visited=bfs(edge_index_list, j)    
        if (len(visited)>1):
            g_reachable_nodes.append(visited)          
    temp.setReachableNodes(g_reachable_nodes)
    
   # calculate weight of graph
    weight=sum(g.edge_attr)
    temp.setWeight(weight)
        
    # calculate number of nodes in each graph
    edge_list=g.edge_index.tolist()    # convert tensor to list    
    edge_list = list(itertools.chain.from_iterable(edge_list))     # it merges list of lists to a single list consisting the nodes , but not unique nodes
    unique_nodes=set(edge_list)
    temp.setUniqueNodes(list(unique_nodes))    
    num_nodes=len(set(unique_nodes))   # number of unique nodes
    #num_nodes=24
    temp.setNumNodes(num_nodes)
        
    # calcuate average connectivity
    adj_mtrx=torch.tensor(24) 
    adj_mtrx=convertToAdjMatrix(g.edge_index).reshape(24,24)

    # calculate average graph connectivity - First we should calculate the maximumdisjoint paths for each pairs of nodes in graph
    graph = Graph(adj_mtrx)                   
    max_disjoint_path=np.zeros((numOfChrom,numOfChrom), dtype=int)

    for k in range (numOfChrom):
        for t in range (numOfChrom):     
            graph_copy=copy.deepcopy(graph)
            if (t!=k) :
                max_disjoint_path[t,k]=graph_copy.findDisjointPaths(t, k);
    temp.setMaxDisjointPath(max_disjoint_path)
    avg_con=sum(sum(max_disjoint_path))
    
    # just for self loop scanrio: if there is only 1 or 2 nodes in the graph, set avg_con to 0 
    if (num_nodes==1 or num_nodes==0):
        avg_con=0
    else:
        avg_con=avg_con/ (math.factorial(num_nodes) / (2 * math.factorial(num_nodes-2)))   
    temp.setAvgCon(avg_con)                               
    data_list_append.append(temp)  
    
# =============================================================================
# # Part 3 : Filterig the graphs based on specified criteria
# =============================================================================
print('Part III: Calculating the criteri and filtering graphs')
# in this section we filter the graphs on data_list_append based on avg_con_threshold to keep most connected graphs for training. 
# First we store the filtered graph into data_list_append_fil. Each element in data_list_append_fil is of type data_list_append. 
#In order to use Dataloader in training section later, we need to have graphs of type Data.  
# so in this section we derive the filtered elements of type Data from data_list and store them into data_list_final
# we also derive the index of each filtered graph from original data_list in case we need the appended information ( avg_con, weight, max_disjoint_path, etc.) from data_list_append

#  data_list_final  : The final graphs ehich meet the criteria and are of type "Data"
# data_list_append  : These are the same graphs from data_list but including extended information such as avg_con, max_disjoint_path, weight, num_nodes, unique_nodes, etc. The elements are of type "Data"
# data_list_append_fil : The graphs which meet the specified criteria , and are of type " DataGraphAppend"
# data_list_index : The indices of the graphs in original data_list and data_list_append. these indices are NOT same in data_list_append_fil. These are the graphs that are chosen as the final results. 


data_list_append_fil=[]
data_list_index=[]   
for i in range (len(data_list_append)): 
    #print ("i:",i," ", data_list_append[i].weight)
    if (data_list_append[i].avg_con>avg_con_threshold):
        data_list_index.append(i)
    
data_list_final=[]
for i in range (len(data_list_index)):
    data_list_final.append(data_list[data_list_index[i]])


# Graph Neural Network implementation
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BatchNorm1d
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv,GATConv
from torch_geometric.nn import ChebConv
from torch_geometric.nn import global_add_pool, global_mean_pool, dense_diff_pool,global_max_pool
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean

class Net(torch.nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATConv(num_node_features,hidden_dim)
        self.conv2 = GATConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.conv6 = GCNConv(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, num_classes)
        


    def forward(self, data):
        x, edge_index,batch_index = data.x, data.edge_index, data.batch             

        x = self.conv1(x, edge_index)         
        x = F.relu(x)                      
        x = self.conv2(x, edge_index)                  
        x = F.relu(x)   
        x = self.conv3(x, edge_index)                    
        x = F.relu(x)       
        x = self.conv4(x, edge_index)                     
        x = F.relu(x)       
        x = self.conv5(x, edge_index)                      
        x = F.relu(x)       
        x = self.conv6(x, edge_index)                       
        x = F.relu(x)
        x = global_max_pool(x, batch_index)          
        #x = self.fc1(x)
        x = F.dropout(x,p=0.2,training=self.training)
        #x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(epoch):
    model.train()
    loss_all = 0
    i=0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)           
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step() 
        i=i+1           # adjust parameters based on the calculated gradients
        losspersample=loss.item()
        
    return loss_all / len(train_dataset)

import seaborn as sn
predictions, targets = [], []
report=[]
np_cm=np.empty([3,3])

def test(loader, cm,report):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
       
        # This is for confusion matrix
        pred = pred.detach().cpu().numpy()  
        labels = (data.y).detach().cpu().numpy()            
        for i in range(len(pred)):
            predictions.append(pred[i])
            targets.append(labels[i]) 
            
    #fpr, tpr,threshold  = roc_curve(targets,predictions)
    np_cm= metrics.confusion_matrix(targets,predictions)
   
    # this is for classification report
    #report.append(classification_report(targets, predictions,output_dict=True))     
    cm.append(np_cm)
    np_cm=metrics.confusion_matrix(targets,predictions)
    norm_mm=(np_cm.T/np_cm.astype(np.float).sum(axis=1)).T
    disp=sklearn.metrics.ConfusionMatrixDisplay(norm_mm,['Breast','Pancreas','Prostate'])
    disp.plot(cmap="OrRd")
    plt.show()      
    return correct / len(loader.dataset)

train_size = int(training_rate * len(data_list_final))
test_size = len(data_list_final) - train_size

y_data=[]
train_dataset, test_dataset = torch.utils.data.random_split(data_list_final, [train_size, test_size])

# Dealing with imbalanced data using WeithedRandomSampler
count0final=0
count1final=0
count2final=0

dataset=train_dataset

for i in range (len(dataset)):
    if dataset[i].y==0 :
        count0final=count0final+1
    elif dataset[i].y==1 :
        count1final=count1final+1
    elif dataset[i].y==2 :
        count2final=count2final+1
        
class_counts = [count0final, count1final,count2final]
num_samples = sum(class_counts)

labels=[]
for i in range (len(train_dataset)):
    labels.append(dataset[i].y)  #corresponding labels of samples

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

train_loader = DataLoader(dataset = train_dataset, batch_size=batch_size,sampler = sampler)   # sampler automatically shuffles!
test_loader = DataLoader(dataset = test_dataset, batch_size=batch_size, shuffle=False)  # Shuffle=False, because order of data doesn't change th emetric

print("Length of the train_loader:", len(train_loader))
print("Length of the val_loader:", len(test_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
hist = {"loss":[], "acc":[], "test_acc":[]}

cm_test=[]
cm_train=[]

report_train=[]
report_test=[]

for epoch in range(1, num_epochs):    
    print('epoch', epoch, '----------------------------------------------------------------------------')
    train_loss = train(epoch)
    train_acc = test(train_loader, cm_train,report_train)  
    test_acc = test(test_loader,cm_test,report_test)
    hist["loss"].append(train_loss)
    hist["acc"].append(train_acc)
    hist["test_acc"].append(test_acc)
    print(f'Epoch: {epoch}, Train loss: {train_loss:.3}, Train_acc: {train_acc:.3}, Test_acc: {test_acc:.3}')

# Plot accuracy and loss 
plt.title('training loss')
plt.ylim([0.0, 1.0])
plt.xlabel('epoch')
plt.plot(hist['loss'])
plt.show()

plt.title('training accuracy')
plt.ylim([0.0, 1.0])
plt.xlabel('epoch')
plt.plot(hist['acc'])
plt.show()

plt.title('testing accuracy')
plt.ylim([0.0, 1.0])
plt.xlabel('epoch')
plt.plot(hist['test_acc'])
plt.show()
print('end')

# saving the model
torch.save(model, 'graphchrom3')

model=torch.load('graphchrom3')