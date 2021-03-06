import math
from matplotlib import pyplot as plt


def main():
    font = {'family': 'SimSun', 'size': '12'}
    plt.rc('font', **font)
    # plt.figure(figsize=(10, 8))
    hits = {
        0.25: {
            0:    '81.49 / 92.23 / 93.99 / 95.15',
            0.25: '83.62 / 92.35 / 94.05 / 95.15',
            0.5:  '82.83 / 92.05 / 93.69 / 94.48',
            0.75: '78.34 / 89.87 / 91.75 / 92.60',
            1.0:  '74.70 / 85.92 / 88.59 / 91.14',
        },
        0.5: {
            0:    '80.40 / 92.35 / 93.57 / 94.48',
            0.25: '82.65 / 92.90 / 94.11 / 95.27',
            0.5:  '81.92 / 92.35 / 94.30 / 95.39',
            0.75: '76.09 / 88.90 / 90.90 / 92.72',
            1.0:  '74.51 / 87.26 / 89.99 / 91.26',
        },
        0.75: {
            0:    '81.25 / 91.20 / 93.14 / 94.78',
            0.25: '81.98 / 92.48 / 93.81 / 94.78',
            0.5:  '82.34 / 92.48 / 93.75 / 94.54',
            0.75: '77.18 / 89.99 / 91.81 / 93.14',
            1.0:  '75.30 / 87.92 / 90.17 / 91.14',
        }
    }
    hit_k = [1, 5, 10, 20]
    # 分别对应于 1, 5, 10, 20
    new_hits = [{} for _ in range(len(hit_k))]
    for first_mk, first_hit in hits.items():
        cur_mks, cur_results_map = [], {}
        for second_mk, second_res in first_hit.items():
            cur_mks.append(second_mk)
            results = [float(item) for item in second_res.split(' / ')]
            assert len(results) == len(hit_k)
            for res, k in zip(results, hit_k):
                cur_results_map.setdefault(k, [])
                cur_results_map[k].append(res)
        for hid in range(len(hit_k)):
            new_hits[hid][first_mk] = (cur_mks, cur_results_map[hit_k[hid]])
    for item in new_hits:
        print(item)
    colors = {
        0.25: 'r',
        0.5:  'g',
        0.75: 'b',
    }
    for hid in range(len(hit_k)):
        plt.subplot(2, 2, hid + 1)
        results = new_hits[hid]
        max_res, min_res = 0, 100
        for first_mk, (second_mks, second_results) in results.items():
            if first_mk != 0.25:
                continue
            plt.plot(second_mks, second_results, linestyle='-', marker='.',
                     color=colors[first_mk], label=f'{first_mk:.2f}')
            max_res = max(max_res, max(second_results))
            min_res = min(min_res, min(second_results))
        plt.xlabel('II.覆盖率')
        plt.ylabel(f'Hit@{hit_k[hid]}')
        plt.xticks(ticks=[0, 0.25, 0.5, 0.75, 1.0])
        max_tick = math.ceil(max_res)
        min_tick = math.floor(min_res)
        plt.yticks(ticks=range(min_tick, max_tick+1, 1 if max_tick - min_tick <= 7 else 2))
        # plt.legend()
        plt.title(f'Hit@{hit_k[hid]}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
