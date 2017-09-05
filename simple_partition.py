import numpy as np
import random as rnd

def part_graph(clusters_num, adjacency):
    nodes_num = len(adjacency)
    #print(nodes_num, clusters_num)
    size1 = nodes_num // clusters_num
    size2 = size1+1
    clusters_num2 = nodes_num % clusters_num
    clusters_num1 = clusters_num - clusters_num2
    #print("{}*{} + {}*{} = {}".format(size1, clusters_num1, size2, clusters_num2,
    #                                  size1*clusters_num1 + size2*clusters_num2))

    partition = np.zeros([nodes_num], dtype = int)
    #print(partition)

    untouched = set(range(nodes_num))
    nb_global = set()
    nb_cur_layer = set()
    nb_next_layer = set()
    used = set()
    node_sets = (untouched, nb_global, nb_cur_layer, nb_next_layer, used)

    node_state = np.zeros([nodes_num], dtype = int)
    UNTOUCHED     = 0
    NB_GLOBAL     = 1
    NB_CUR_LAYER  = 2
    NB_NEXT_LAYER = 3
    USED          = 4

    def change_node_state(node, new_state):
        ori_state = node_state[node]
        node_sets[ori_state].remove(node)
        node_sets[new_state].add(node)
        node_state[node] = new_state

    def change_state_all(ori_state, new_state):
        ori_layer = node_sets[ori_state]
        for node in ori_layer:
            node_state[node] = new_state

        node_sets[new_state].update(ori_layer)
        node_sets[ori_state].clear()

    cur_size = 0
    cluster_index = 0
    cluster_size = size2
    if cluster_index == clusters_num2: cluster_size = size1

    while cluster_index < clusters_num:

        if len(nb_cur_layer) > 0:
            cur_set = nb_cur_layer
            #print("nb_cur_layer")
        elif len(nb_next_layer) > 0:
            change_state_all(NB_NEXT_LAYER, NB_CUR_LAYER)
            #print("nb_next_layer")
            cur_set = nb_cur_layer
        elif len(nb_global) > 0:
            cur_set = nb_global
            #print("nb_global")
        else:
            cur_set = untouched
            #print("untouched")

        #node = rnd.choice(tuple(cur_set))
        node = cur_set.pop()
        node_state[node] = USED
        partition[node] = cluster_index
        #change_node_state(node, USED)
        cur_size += 1

        for neighbor in adjacency[node]:
            if node_state[neighbor] < NB_CUR_LAYER:
                change_node_state(neighbor, NB_NEXT_LAYER)


        if cur_size == cluster_size:
            change_state_all(NB_CUR_LAYER, NB_GLOBAL)
            change_state_all(NB_NEXT_LAYER, NB_GLOBAL)

            cluster_index += 1
            if cluster_index == clusters_num2: cluster_size = size1
            cur_size = 0

    return partition

if __name__ == "__main__":
    
    num_clusters = 35
    adjacency = [np.array([112,   1]), np.array([  0, 112,   2,   3]), np.array([1, 3]), np.array([1, 2, 4, 5]), np.array([3, 5]), np.array([3, 4, 6, 7]), np.array([5, 7]), np.array([8, 9, 5, 6]), np.array([9, 7]), np.array([ 8, 10, 11,  7]), np.array([ 9, 11]), np.array([ 9, 10, 12, 13]), np.array([11, 13]), np.array([11, 12, 14, 15]), np.array([13, 15]), np.array([16, 17, 13, 14]), np.array([17, 15]), np.array([16, 18, 19, 15]), np.array([17, 19]), np.array([17, 18, 20, 77]), np.array([19, 21, 22, 77]), np.array([20, 22]), np.array([20, 21, 30, 23]), np.array([24, 25, 22, 30]), np.array([25, 23]), np.array([24, 26, 29, 23]), np.array([25, 27, 28, 29]), np.array([26, 28]), np.array([26, 27]), np.array([25, 26]), np.array([22, 23, 38, 31]), np.array([32, 33, 30, 38]), np.array([33, 31]), np.array([32, 34, 37, 31]), np.array([33, 35, 36, 37]), np.array([34, 36]), np.array([34, 35]), np.array([33, 34]), np.array([31, 50, 30, 39]), np.array([40, 41, 50, 38]), np.array([41, 39]), np.array([40, 49, 42, 39]), np.array([41, 49, 43, 44]), np.array([42, 44]), np.array([42, 43, 45, 46]), np.array([44, 46]), np.array([48, 44, 45, 47]), np.array([48, 46]), np.array([46, 47]), np.array([41, 42]), np.array([39, 51, 62, 38]), np.array([50, 52, 53, 62]), np.array([51, 53]), np.array([51, 52, 61, 54]), np.array([56, 53, 61, 55]), np.array([56, 54]), np.array([57, 58, 54, 55]), np.array([56, 58]), np.array([56, 57, 59, 60]), np.array([58, 60]), np.array([58, 59]), np.array([53, 54]), np.array([50, 51, 70, 63]), np.array([64, 65, 62, 70]), np.array([65, 63]), np.array([64, 66, 69, 63]), np.array([65, 67, 68, 69]), np.array([66, 68]), np.array([66, 67]), np.array([65, 66]), np.array([72, 63, 62, 71]), np.array([72, 70]), np.array([73, 74, 70, 71]), np.array([72, 74]), np.array([72, 73, 75, 76]), np.array([74, 76]), np.array([74, 75]), np.array([19, 20, 78, 95]), np.array([80, 95, 77, 79]), np.array([80, 78]), np.array([81, 82, 78, 79]), np.array([80, 82]), np.array([80, 81, 90, 83]), np.array([82, 90, 84, 85]), np.array([83, 85]), np.array([89, 83, 84, 86]), np.array([88, 89, 85, 87]), np.array([88, 86]), np.array([86, 87]), np.array([85, 86]), np.array([82, 91, 83, 94]), np.array([90, 92, 93, 94]), np.array([91, 93]), np.array([91, 92]), np.array([90, 91]), np.array([ 96, 105,  77,  78]), np.array([ 97,  98, 105,  95]), np.array([96, 98]), np.array([ 96,  97,  99, 102]), np.array([ 98, 100, 101, 102]), np.array([ 99, 101]), np.array([ 99, 100]), np.array([104,  98,  99, 103]), np.array([104, 102]), np.array([102, 103]), np.array([ 96, 106, 109,  95]), np.array([105, 107, 108, 109]), np.array([106, 108]), np.array([106, 107]), np.array([105, 106, 110, 111]), np.array([109, 111]), np.array([109, 110]), np.array([  0, 113, 130,   1]), np.array([112, 114, 115, 130]), np.array([113, 115]), np.array([113, 114, 123, 116]), np.array([115, 123, 117, 118]), np.array([116, 118]), np.array([122, 116, 117, 119]), np.array([120, 121, 122, 118]), np.array([121, 119]), np.array([120, 119]), np.array([118, 119]), np.array([115, 124, 125, 116]), np.array([123, 125]), np.array([123, 124, 126, 127]), np.array([125, 127]), np.array([128, 129, 125, 126]), np.array([129, 127]), np.array([128, 127]), np.array([112, 113, 131, 142]), np.array([130, 132, 133, 142]), np.array([131, 133]), np.array([131, 132, 141, 134]), np.array([136, 133, 141, 135]), np.array([136, 134]), np.array([137, 138, 134, 135]), np.array([136, 138]), np.array([136, 137, 139, 140]), np.array([138, 140]), np.array([138, 139]), np.array([133, 134]), np.array([154, 130, 131, 143]), np.array([144, 145, 154, 142]), np.array([145, 143]), np.array([144, 153, 146, 143]), np.array([145, 153, 147, 148]), np.array([146, 148]), np.array([146, 147, 149, 150]), np.array([148, 150]), np.array([152, 148, 149, 151]), np.array([152, 150]), np.array([150, 151]), np.array([145, 146]), np.array([162, 155, 142, 143]), np.array([154, 162, 156, 157]), np.array([155, 157]), np.array([161, 155, 156, 158]), np.array([160, 161, 157, 159]), np.array([160, 158]), np.array([158, 159]), np.array([157, 158]), np.array([170, 163, 154, 155]), np.array([162, 170, 164, 165]), np.array([163, 165]), np.array([169, 163, 164, 166]), np.array([168, 169, 165, 167]), np.array([168, 166]), np.array([166, 167]), np.array([165, 166]), np.array([162, 171, 188, 163]), np.array([170, 172, 173, 188]), np.array([171, 173]), np.array([171, 172, 174, 175]), np.array([173, 175]), np.array([176, 173, 174, 183]), np.array([183, 177, 178, 175]), np.array([176, 178]), np.array([176, 177, 179, 182]), np.array([178, 180, 181, 182]), np.array([179, 181]), np.array([179, 180]), np.array([178, 179]), np.array([184, 176, 187, 175]), np.array([185, 186, 187, 183]), np.array([184, 186]), np.array([184, 185]), np.array([184, 183]), np.array([170, 171, 196, 189]), np.array([188, 196, 190, 191]), np.array([189, 191]), np.array([192, 195, 189, 190]), np.array([193, 194, 195, 191]), np.array([192, 194]), np.array([192, 193]), np.array([192, 191]), np.array([202, 188, 197, 189]), np.array([202, 196, 198, 199]), np.array([197, 199]), np.array([200, 201, 197, 198]), np.array([201, 199]), np.array([200, 199]), np.array([203, 204, 196, 197]), np.array([202, 204]), np.array([202, 203])]

    part_graph(num_clusters, adjacency)
