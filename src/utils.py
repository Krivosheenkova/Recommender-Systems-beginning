# utils.py

import numpy as np

def prefilter_items(data, item_features, take_n_popular=5000, fake_id=99999):

    print('== Starting prefilter info ==')
    n_users = data.user_id.nunique()
    n_items = data.item_id.nunique()
    sparsity = float(data.shape[0]) / float(n_users*n_items) * 100
    print('shape: {}'.format(data.shape))
    print('# users: {}'.format(n_users))
    print('# items: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    start_columns = set(data.columns.tolist())
    data_train = data.copy()

    # do not use top popular items (they'd be bought anyway)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index()
    popularity['user_id'] = popularity['user_id'] / data_train.user_id.nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    top_popular = popularity[popularity['share_unique_users'] > .5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]

    # do not use top not popular
    top_not_popular = popularity[popularity.share_unique_users < .0009].item_id.tolist()
    data = data[~data.item_id.isin(top_not_popular)]

    # do not use items that have not been sold in the last 12 month
    num_weeks = 12*4
    start_week = data_train.week_no.max() - num_weeks
    items_sold_last_year = data[data.week_no >= start_week].item_id.tolist()
    data = data[data.item_id.isin(items_sold_last_year)]

    # do not use not popular departments
    merged_data_departments = data_train[['user_id', 'item_id', 'quantity']].merge(item_features[['item_id', 'department']], how='left')
    quantity_by_department = merged_data_departments.groupby('department')['quantity'].sum().reset_index()
    quantity_by_department['coef'] = quantity_by_department.quantity / quantity_by_department.quantity.sum()
    not_popular_departments = quantity_by_department[quantity_by_department.coef < quantity_by_department.coef.quantile(0.25)].department.tolist()
    
    not_popular_departments_items = item_features[
        item_features.department.isin(not_popular_departments)].item_id.tolist()
    data = data[~data.item_id.isin(not_popular_departments_items)]

    # do not use too expensive and too cheap items
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    high_cost_threshold = 2  # mailing cost 2$ 
    low_cost_threshold = data_train.sales_value.quantile(.11)
    data = data[
        (data.sales_value < high_cost_threshold)
        &
        (data.sales_value > low_cost_threshold)
        ]    

    # do not use too popular stores
    store_df = data.groupby('store_id')['user_id'].nunique().reset_index()
    data = data[~data.store_id.isin(
        store_df[store_df.user_id > store_df.user_id.quantile(.985)].store_id.tolist()
    )]

    # Take n top popularity
    popularity = data.groupby('item_id')['quantity'].sum().reset_index(name='n_sold')
    # popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()
    
    # Insert fake item_id, if user have bought from top then user have been "bought" this item already
    data.loc[~data['item_id'].isin(top), 'item_id'] = fake_id

    # take n poplar items
    if take_n_popular:
        popularity = data.groupby('item_id')['user_id'].nunique().reset_index().sort_values('user_id', ascending=False).item_id.tolist()
        data = data[data.item_id.isin(popularity[:take_n_popular])]
    

    print('== Ending prefilter info ==')
    print('shape: {}'.format(data.shape))
    n_users = data.user_id.nunique()
    n_items = data.item_id.nunique()
    sparsity = float(data.shape[0]) / float(n_users*n_items) * 100
    print('# users: {}'.format(n_users))
    print('# items: {}'.format(n_items))
    print('Sparsity: {:4.3f}%'.format(sparsity))
    end_columns = set(data.columns.tolist())
    print(bold('new_columns:'), 
          end_columns-start_columns)

    return data

def bold(string: str) -> str:
    """returns bold string"""
    return '\033[1m'+string+'\033[0m'