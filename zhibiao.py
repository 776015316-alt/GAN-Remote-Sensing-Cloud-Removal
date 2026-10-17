import pandas as pd
import itertools
import networkx as nx

# 读取数据
df = pd.read_excel("C:\\Users\\Administrator\\Desktop\\美国2010-2014合作专利.xlsx")
applicants_series = df['申请人 (Cleaned)']

# 创建无向图
G = nx.Graph()

# 构建合作网络
for applicants in applicants_series.dropna():
    nodes = [node.strip() for node in applicants.split('|') if node.strip()]
    # 添加所有可能的合作边（全连接）
    for u, v in itertools.combinations(nodes, 2):
        G.add_edge(u, v)

# 计算指标
n = G.number_of_nodes()

# 度中心度（已自动除以n-1）
degree_cent = nx.degree_centrality(G)

# 中间中心度（不进行归一化）
between_cent = nx.betweenness_centrality(G, normalized=False)

# 接近中心度
close_cent = nx.closeness_centrality(G)

# 个体网络规模（节点度数）
ego_size = dict(G.degree())

# 整理结果
results = []
for node in G.nodes():
    results.append({
        '申请人': node,
        '度中心度': round(degree_cent[node], 6),
        '中间中心度': round(between_cent[node], 6),
        '接近中心度': round(close_cent[node], 6),
        '个体网络规模': ego_size[node]
    })

# 创建结果DataFrame并保存
result_df = pd.DataFrame(results)
result_df.to_excel('1011.xlsx', index=False)

print("计算完成，结果已保存到network_metrics.xlsx")