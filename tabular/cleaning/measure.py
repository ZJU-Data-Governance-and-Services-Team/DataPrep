# Precision = TP / (TP + FP)
# Recall = TP / (TP + FN)
# F1 Score = 2 * Pre * Rec / (Pre + Rec)
# pre = correctly_repaired / all_repaired
# rec = correctly_repaired / all_need_repair
# f1 = 1 / ((1 / pre + 1 / rec) / 2) = 2 * pre * rec / (pre + rec)
import pandas as pd
import re


# Measure repair precision, recall, and F1 against clean data.
def measure_repair(clean_path, dirty_path_ori, res_path):
    df_clean = pd.read_csv(clean_path, dtype=str).fillna('nan')
    df_dirty = pd.read_csv(dirty_path_ori, dtype=str).fillna('nan')
    df_cleaned = pd.read_csv(res_path, dtype=str).fillna('nan')
    results = ''

    wrong_2_right = 0
    right_2_wrong = 0
    wrong_2_wrong = 0
    wrong_not_change = 0
    all_repaired = 0
    all_need_repair = 0
    for index, row in df_clean.iterrows():
        for i in range(len(df_clean.columns)):
            if str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i]):
                all_need_repair += 1
            if str(df_dirty.iat[index, i]) != str(df_cleaned.iat[index, i]):
                all_repaired += 1
                # right repair
                if str(df_cleaned.iat[index, i]) == str(df_clean.iat[index, i]):
                    wrong_2_right += 1
                # wrong repair
                if str(df_cleaned.iat[index, i]) != str(df_clean.iat[index, i]):
                    if str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i]):
                        wrong_2_wrong += 1
                    if str(df_dirty.iat[index, i]) == str(df_clean.iat[index, i]):
                        right_2_wrong += 1
            # error not repair
            if str(df_dirty.iat[index, i]) == str(df_cleaned.iat[index, i]):
                if str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i]):
                    wrong_not_change += 1


    pre = wrong_2_right / (all_repaired+1e-8)
    rec = wrong_2_right / (all_need_repair+1e-8)
    f1 = 2 * pre * rec / (pre+rec+1e-8)
    # print('all_wrong_num:', all_need_repair)
    # print('all_repaired_num:', all_repaired)
    # print('wrong_2_right:', wrong_2_right)
    # print('wrong_2_wrong:', wrong_2_wrong)
    # print('wrong_not_change:', wrong_not_change)
    # print('right_2_wrong:', right_2_wrong)
    # print('pre:', pre)
    # print('rec:', rec)
    # print('f1:', f1)

    results += 'all_wrong_num:' + str(all_need_repair) + '\n'
    results += 'all_repaired_num:' + str(all_repaired) + '\n'
    results += 'wrong_2_right:' + str(wrong_2_right) + '\n'
    results += 'wrong_2_wrong:' + str(wrong_2_wrong) + '\n'
    results += 'wrong_not_change:' + str(wrong_not_change) + '\n'
    results += 'right_2_wrong:' + str(right_2_wrong) + '\n'
    results += 'pre:' + str(pre) + '\n'
    results += 'rec:' + str(rec) + '\n'
    results += 'f1:' + str(f1) + '\n'
    
    print(f'f1: {f1}')

    # print('\n')
    results += '\n'
    # print('wrong_2_right:')
    results += 'wrong_2_right:' + '\n'
    for index, row in df_clean.iterrows():
        label = 0
        for i in range(len(df_clean.columns)):
            if str(df_dirty.iat[index, i]) != str(df_cleaned.iat[index, i]):
                if str(df_cleaned.iat[index, i]) == str(df_clean.iat[index, i]):
                    if label == 0:
                        formatted_data = {}
                        for col in df_clean.columns:
                            formatted_data[col] = row[col]
                        formatted_row = "{" + "; ".join(
                            [f"{key}: {value}" for key, value in formatted_data.items()]) + "}"
                        label = 1
                        # print(formatted_row)
                        results += formatted_row + '\n'
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_clean.iat[index, i]), '(clean)')
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_dirty.iat[index, i]), '(dirty)')
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_cleaned.iat[index, i]), '(cleaned)')
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(df_clean.iat[index, i]) + ' (clean)\n'
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(df_dirty.iat[index, i]) + ' (dirty)\n'
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(df_cleaned.iat[index, i]) + ' (cleaned)\n'

    # print('\n')
    results += '\n'
    # print('wrong_2_wrong:')
    results += 'wrong_2_wrong:' + '\n'
    for index, row in df_clean.iterrows():
        label = 0
        for i in range(len(df_clean.columns)):
            if str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i]):
                if str(df_dirty.iat[index, i]) != str(df_cleaned.iat[index, i]):
                    if str(df_cleaned.iat[index, i]) != str(df_clean.iat[index, i]):
                        if label == 0:
                            formatted_data = {}
                            for col in df_clean.columns:
                                formatted_data[col] = row[col]
                            formatted_row = "{" + "; ".join(
                                [f"{key}: {value}" for key, value in formatted_data.items()]) + "}"
                            label = 1
                            # print(formatted_row)
                            results += formatted_row + '\n'
                        # print(str(index), ',', df_clean.columns[i], ':', str(df_clean.iat[index, i]), '(clean)')
                        # print(str(index), ',', df_clean.columns[i], ':', str(df_dirty.iat[index, i]), '(dirty)')
                        # print(str(index), ',', df_clean.columns[i], ':', str(df_cleaned.iat[index, i]), '(cleaned)')
                        results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                            df_clean.iat[index, i]) + ' (clean)\n'
                        results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                            df_dirty.iat[index, i]) + ' (dirty)\n'
                        results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                            df_cleaned.iat[index, i]) + ' (cleaned)\n'

    # print('\n')
    results += '\n'
    # print('wrong_not_change:')
    results += 'wrong_not_change:' + '\n'
    for index, row in df_clean.iterrows():
        label = 0
        for i in range(len(df_clean.columns)):
            if str(df_dirty.iat[index, i]) != str(df_clean.iat[index, i]):
                if str(df_cleaned.iat[index, i]) == str(df_dirty.iat[index, i]):
                    if label == 0:
                        formatted_data = {}
                        for col in df_clean.columns:
                            formatted_data[col] = row[col]
                        formatted_row = "{" + "; ".join(
                            [f"{key}: {value}" for key, value in formatted_data.items()]) + "}"
                        label = 1
                        # print(formatted_row)
                        results += formatted_row + '\n'
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_clean.iat[index, i]), '(clean)')
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_dirty.iat[index, i]), '(dirty)')
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_cleaned.iat[index, i]), '(cleaned)')
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                        df_clean.iat[index, i]) + ' (clean)\n'
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                        df_dirty.iat[index, i]) + ' (dirty)\n'
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                        df_cleaned.iat[index, i]) + ' (cleaned)\n'

    # print('\n')
    results += '\n'
    # print('right_2_wrong:')
    results += 'right_2_wrong:' + '\n'
    for index, row in df_clean.iterrows():
        label = 0
        for i in range(len(df_clean.columns)):
            if str(df_dirty.iat[index, i]) == str(df_clean.iat[index, i]):
                if str(df_cleaned.iat[index, i]) != str(df_clean.iat[index, i]):
                    if label == 0:
                        formatted_data = {}
                        for col in df_clean.columns:
                            formatted_data[col] = row[col]
                        formatted_row = "{" + "; ".join(
                            [f"{key}: {value}" for key, value in formatted_data.items()]) + "}"
                        label = 1
                        # print(formatted_row)
                        results += formatted_row + '\n'
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_clean.iat[index, i]), '(clean)')
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_dirty.iat[index, i]), '(dirty)')
                    # print(str(index), ',', df_clean.columns[i], ':', str(df_cleaned.iat[index, i]), '(cleaned)')
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                        df_clean.iat[index, i]) + ' (clean)\n'
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                        df_dirty.iat[index, i]) + ' (dirty)\n'
                    results += str(index) + ', ' + df_clean.columns[i] + ': ' + str(
                        df_cleaned.iat[index, i]) + ' (cleaned)\n'

    res_path = res_path.replace('.csv', '.txt')
    with open(res_path, "w") as file:
        file.write(results)
    file.close()


if __name__ == '__main__':
    pass