#  @file: ahplot.py
#  @version：1.0.5
#  @brief: AHP Hierarchy Model Plotting
#  @creation date: 2025.10.11
#  @last modified date: 2025.11.11 
#  @authors: S. Yang
#  @copyright: © 2025 S. Yang. All rights reserved.
#  @license: This program is licensed under the MIT license. 



import matplotlib.pyplot as plt
import networkx as nx

graph_color = '#4086e8'
node_color = "#8f7b67"
label_color = "#121010"
node_options = {'node_color': node_color,
                'node_shape': 's', 
                'node_size':0,
                'bbox':dict(facecolor = "#eecece"),
                'font_color': label_color}
edge_options = {'arrowstyle': '-|>', 
                'arrowsize': 10, 
                'width': 1,
                'edge_color': graph_color}


def draw_ahp_hierarchy(goal, criteria, alternatives, judgments):
    # 创建有向图
    G = nx.DiGraph()    
    # 添加总目标节点
    G.add_node(goal, type='goal')    
    # 添加准则节点
    for c in criteria:
        G.add_node(c, type='criteria')    
    # 添加备选方案节点
    for a in alternatives:
        G.add_node(a, type='alternative')    
    # 添加总目标到准则的边
    for i, c in enumerate(criteria):
        G.add_edge(goal, c, weight=1)
    # 添加准则到备选方案的边
    for i, c in enumerate(criteria):
        for j, a in enumerate(alternatives):
            G.add_edge(c, a, weight=judgments[i][j])
    
    # 设置布局和节点样式
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')  # 'neato', 'dot', 'twopi', 'circo', 'fdp', 'nop' 'sfdp'
    # pos = nx.spring_layout(G)  # 随机布局
    # 控制节点的顺序
    # pos = {}
    middle = 280.0
    layer0 = 160.0
    layer1 = 99.0
    layer2 = 15.0
    count = 0
    for p in pos.keys():
        if count == 0:
            pos[p] = (middle, layer0)
        if count > 0 and count <= 4:
            pos[p] = (110*count, layer1)
        if count > 4:
            pos[p] = (110*(count-4), layer2)
        count += 1

    # 绘制图
    nx.draw(G, pos, with_labels=True, 
            **edge_options,
            **node_options)
    
    # print(len(G.edges(data=True)))
    # print(G.nodes(), G.edges())
    # edge_labels = dict([((u, v), d['weight']) for (u, v, d) in G.edges(data=True)])
    # edge_labels = {}
    # for (u, v, d) in G.edges(data=True):
    #      edge_labels[(u, v)]=d['weight']
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title('AHP Hierarchy Model')
    plt.show()

# 数据
goal = 'Cultural Difference' 
criteria = ['indep./interdep.', 'indiv./collect.', 'tightness/losseness', 'relational mobility']
alternatives = ['Historical Onto.', 'Aesthetic Onto.', 'Semiotic Onto.', 'Socialogical Onto.']
judgments = [
    [1, 1, 1, 1],  # Historical vs Criterion 1, Aesthetic vs Criterion 1, Semiotic vs Criterion 1, Socialogical vs Criterion 1
    [3, 1, 1, 1],  # Historical vs Criterion 2, Aesthetic vs Criterion 2, Semiotic vs Criterion 2, Socialogical vs Criterion 2
    [1, 1, 1, 1],  # Historical vs Criterion 3, Aesthetic vs Criterion 3, Semiotic vs Criterion 3, Socialogical vs Criterion 3
    [1, 1, 3, 1]   # Historical vs Criterion 4, Aesthetic vs Criterion 4, Semiotic vs Criterion 4, Socialogical vs Criterion 4
]
draw_ahp_hierarchy(goal, criteria, alternatives, judgments)
