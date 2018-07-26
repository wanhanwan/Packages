from QuantLib.utils import (DropOutlier,
                            NonLinearSize,
                            Standard,
                            Orthogonalize,
                            Fillna_Barra,
                            Join_Factors)
import pandas as pd


class CSOperator(object):
    def __init__(self, data_source, risk_db):
        self.data_source = data_source
        self.riskdb = risk_db
        self.cached_data = {}
        self.stock_idx = None

    def set_dimension(self, idx):
        self.stock_idx = idx
        self.all_ids = idx.index.get_level_values(1).unique().tolist()
        self.all_dates = idx.index.get_level_values(0).unique().tolist()

    def set_factor_table(self, factor_table):
        self.data_source.prepare_factor(factor_table)

    def prepare_data(self):
        all_factors = self.data_source.factor_names
        for factor in all_factors:
            factor_data = self.data_source.get_factor_data(factor_name=factor, ids=self.all_ids,
                                                           dates=self.all_dates)
            self.cached_data[factor] = factor_data.reindex(self.stock_idx.index)

    def drop_outlier(self, **kwargs):
        for factor in kwargs['descriptors']:
            new_data = DropOutlier(self.cached_data[factor].reset_index(), factor_name=factor, method=kwargs['method'],
                                   drop_ratio=kwargs['drop_ratio'], drop_mode=kwargs['drop_mode'],
                                   alpha=kwargs['alpha'])[[factor+'_after_drop_outlier']]
            new_data.columns = [factor]
            self.cached_data[factor] = new_data

    def standard(self, **kwargs):
        for factor in kwargs['descriptors']:
            data = self.cached_data[factor]
            mean_weight = kwargs['mean_weight']
            std_weight = kwargs['std_weight']
            if mean_weight is not None:
                if mean_weight == factor:
                    new_mean_weight = mean_weight+'_1'
                    data = pd.concat([data, self.cached_data[mean_weight].rename(columns={mean_weight:new_mean_weight})], axis=1)
                    mean_weight = new_mean_weight
                else:
                    data = pd.concat([data, self.cached_data[mean_weight]], axis=1)
            if kwargs['std_weight'] is not None:
                if std_weight == factor:
                    new_std_weight = std_weight+'_1'
                    data = pd.concat([data, self.cached_data[std_weight].rename(columns={std_weight:new_std_weight})], axis=1)
                    std_weight = new_mean_weight
                else:
                    data = pd.concat([data, self.cached_data[std_weight]], axis=1)
            new_data = Standard(data.reset_index(), factor_name=factor, mean_weight=mean_weight,
                                std_weight=std_weight)[[factor+'_after_standard']]
            new_data.columns = [factor]
            self.cached_data[factor] = new_data

    def orthogonalize(self, **kwargs):
        independents = pd.concat([self.cached_data[x] for x in kwargs['independents']], axis=1)
        new_data = Orthogonalize(self.cached_data[kwargs['dependent']], independents, left_name=kwargs['dependent'],
                                 right_name=kwargs['independents'])
        new_data.columns = [kwargs['dependent']]
        self.cached_data[kwargs['dependent']] = new_data

    def fillna_barra(self, **kwargs):
        all_factors = kwargs['descriptors'] + kwargs['classify'] + kwargs['refs']
        data = pd.concat([self.cached_data[x] for x in all_factors], axis=1)
        new_data = Fillna_Barra(data, factor_names=kwargs['descriptors'], ref_name=kwargs['refs'],
                                classify_name=kwargs['classify'])
        for factor in kwargs['descriptors']:
            self.cached_data[factor] = new_data[[factor]]

    def merge(self, **kwargs):
        to_merge = pd.concat([self.cached_data[x] for x in kwargs['descriptors']], axis=1)
        new_data = Join_Factors(to_merge, kwargs['descriptors'], new_name=kwargs['new_factor'], weight=kwargs['weight'])
        self.cached_data[kwargs['new_factor']] = new_data

    def nonlinear_size(self, **kwargs):
        new_data = NonLinearSize(self.cached_data[kwargs['size']], factor_name=kwargs['size'], new_name=kwargs['new_name'])
        new_data = new_data.to_frame(kwargs['new_name'])
        self.cached_data[kwargs['new_name']] = new_data

    def save_factor(self, save_info):
        save_path = save_info['factor_save_path']
        for factor_name in save_info['factor2save']:
            self.data_source.save_factor(self.cached_data[factor_name], save_path)


FunctionModule = {}
FunctionModule['drop_outlier'] = CSOperator.drop_outlier
FunctionModule['standard'] = CSOperator.standard
FunctionModule['orthogonalize'] = CSOperator.orthogonalize
FunctionModule['fillna_barra'] = CSOperator.fillna_barra
FunctionModule['merge'] = CSOperator.merge
FunctionModule['nonlinear_size'] = CSOperator.nonlinear_size



