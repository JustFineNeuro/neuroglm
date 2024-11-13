import pandas as pd
import numpy as np
import scipy as sp
import patsy
import jax.numpy as jnp
from scipy.linalg import hankel
from GLM import utils as ut


def pac_dummy_dsgn(Xin,nbases=None):
    if nbases == None:
        nbases = 7
    # Categorical variable dummy coded (reward level)
    cat_basis=patsy.dmatrix("C(relvalue, Treatment(reference=1))",{"relvalue":Xin['relvalue']},return_type="dataframe").drop(columns={'Intercept'})

    cat_beta_names = ['cat_2','cat_3','cat_4','cat_5']
    cat_basis = jnp.array(cat_basis)

    ## univariate
    #/For normal basis
    basis_x1 = patsy.dmatrix("cr(x1, df=nbases) - 1", {"x1": Xin['selfspeedmag'],"nbases":nbases}, return_type="dataframe")
    basis_x2 = patsy.dmatrix("cr(x2, df=nbases) - 1", {"x2": Xin['reldist'],"nbases":nbases}, return_type="dataframe")
    basis_x3 = patsy.dmatrix("cr(x3, df=nbases) - 1", {"x3": Xin['wt'],"nbases":nbases}, return_type="dataframe")

    ## Continuous x categorical dummy interactions
    interx1_dfs=[]
    interx2_dfs=[]
    interx3_dfs=[]

    for i in range(cat_basis.shape[1]):
        interx1_dfs.append(pd.DataFrame(np.array(basis_x1) * np.tile(cat_basis[:,i],[nbases,1]).transpose()))
        interx2_dfs.append(pd.DataFrame(np.array(basis_x2) * np.tile(cat_basis[:,i],[nbases,1]).transpose()))
        interx3_dfs.append(pd.DataFrame(np.array(basis_x3) * np.tile(cat_basis[:,i],[nbases,1]).transpose()))

    # List of basis matrices and penalty matrices for each variable
    basis_x_list = [jnp.array(basis_x1.values), jnp.array(basis_x2.values), jnp.array(basis_x3.values)]

    # For dummy coded interactions
    for i in range(len(interx1_dfs)):
        basis_x_list.append(jnp.array(interx1_dfs[i]))
    for i in range(len(interx2_dfs)):
        basis_x_list.append(jnp.array(interx2_dfs[i]))
    for i in range(len(interx3_dfs)):
        basis_x_list.append(jnp.array(interx3_dfs[i]))

    # Construct second-order difference matrices (D) and penalty matrices (S)
    S_list = []

    S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x1, basis_x2=None, is_tensor=False)))
    S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x2, basis_x2=None, is_tensor=False)))
    S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x3, basis_x2=None, is_tensor=False)))

    beta_x_names = ['beta_x1', 'beta_x2', 'beta_x3', 'x1_cat_2', 'x1_cat_3', 'x1_cat_4',
                    'x1_cat_5', 'x2_cat_2', 'x2_cat_3', 'x2_cat_4',
                    'x2_cat_5', 'x3_cat_2', 'x3_cat_3', 'x3_cat_4',
                    'x3_cat_5']


    # Construct cont x cont interactions
    tensor_basis = patsy.dmatrix("te(cr(x2, df=nbases) ,cr(x3, df=nbases) ) - 1", {"x2":Xin['reldist'],"x3": Xin['wt'],
   "nbases":nbases}, return_type="dataframe")

    tensor_basis =[jnp.array(tensor_basis.values)]
    tensor_S = ut.smoothing_penalty_matrix(basis_x2,basis_x3,is_tensor=True)

    return basis_x_list, S_list, tensor_basis, tensor_S, cat_basis, beta_x_names, cat_beta_names



def pac_onehot_dsgn(Xin,nbases=None):
    '''
     # NOTE: we dont include the main effect because it's noninterpretable
     (it woudl be the all zero condition for rewad levels, unobserved)

    :param Xin:
    :param nbases:
    :return:
    '''


    basis_x_list = []
    S_list = []

    if nbases == None:
        nbases = 7

    #Make categorcial felative value code
    cat_basis = patsy.dmatrix("C(relvalue)-1", {"relvalue": Xin['relvalue']}, return_type="dataframe")
    cat_basis = jnp.array(cat_basis)

    cat_beta_names = ['cat_1','cat_2','cat_3','cat_4','cat_5']

    #Only used for tensor
    basis_x2 = patsy.dmatrix("cr(x2,df=nbases) - 1", {"x2": Xin['reldist'], "nbases": nbases}, return_type="dataframe")
    basis_x3 = patsy.dmatrix("cr(x3, df=nbases) - 1", {"x3": Xin['wt'], "nbases": nbases}, return_type="dataframe")

    #Make continuous by cateogrical one hot interaction terms
    basis_inter_x1val = patsy.dmatrix("bs(x1, degree=3,df=7):C(relvalue) - 1", {"x1": Xin['selfspeedmag'],"relvalue":Xin['relvalue']}, return_type="dataframe")
    basis_inter_x2val = patsy.dmatrix("bs(x2, degree=3,df=7):C(relvalue) - 1", {"x2": Xin['reldist'],"relvalue":Xin['relvalue']}, return_type="dataframe")
    basis_inter_x3val = patsy.dmatrix("bs(x3, degree=3,df=7):C(relvalue) - 1", {"x3": Xin['wt'],"relvalue":Xin['relvalue']}, return_type="dataframe")


    # Create a dictionary to store separate dataframes for each unique group number of an interaction
    interx1_dfs = ut.separate_basis_interactions(basis_inter_x1val)
    interx2_dfs = ut.separate_basis_interactions(basis_inter_x2val)
    interx3_dfs = ut.separate_basis_interactions(basis_inter_x3val)

    for keys in interx1_dfs.keys():
        basis_x_list.append(jnp.array(interx1_dfs[keys]))
    for keys in interx2_dfs.keys():
        basis_x_list.append(jnp.array(interx2_dfs[keys]))
    for keys in interx3_dfs.keys():
        basis_x_list.append(jnp.array(interx3_dfs[keys]))

    # Construct second-order difference matrices (D) and penalty matrices (S)

    #Add in cat x cont S_list
    for keys in interx1_dfs.keys():
        S_list.append(jnp.array(ut.smoothing_penalty_matrix(interx1_dfs[keys],basis_x2=None,is_tensor=False)))

    for keys in interx2_dfs.keys():
        S_list.append(jnp.array(ut.smoothing_penalty_matrix(interx2_dfs[keys],basis_x2=None,is_tensor=False)))

    for keys in interx3_dfs.keys():
        S_list.append(jnp.array(ut.smoothing_penalty_matrix(interx3_dfs[keys],basis_x2=None,is_tensor=False)))

    beta_x_names = ['x1_cat_1','x1_cat_2','x1_cat_3','x1_cat_4',
                  'x1_cat_5','x2_cat_1','x2_cat_2','x2_cat_3','x2_cat_4',
                  'x2_cat_5','x3_cat_1','x3_cat_2','x3_cat_3','x3_cat_4',
                  'x3_cat_5']


    #Construct cont x cont tensor

    tensor_basis = patsy.dmatrix("te(cr(x2, df=nbases) ,cr(x3, df=nbases) ) - 1",
                                 {"x2": Xin['reldist'], "x3": Xin['wt'],
                                  "nbases": nbases}, return_type="dataframe")
    tensor_basis = [jnp.array(tensor_basis.values)]
    tensor_S = ut.smoothing_penalty_matrix(basis_x2, basis_x3, is_tensor=True)

    return basis_x_list, S_list, tensor_basis, tensor_S,cat_basis, beta_x_names, cat_beta_names


def pac_cont_dsgn(Xin,params={'basismain':'cr','nbases':9,'basistypeval':'cr','nbasis':5}):
    '''

    :param Xin:
    :param nbases:
    :param contintertype: 'basis' or 'linear'
    :return:
    '''
    if params['nbases'] == None:
        params['nbases'] = 7

    ## univariate
    #/For normal basis
    if params['basismain']=='cr':
        basis_x1 = patsy.dmatrix("cr(x1, df=nbases) - 1", {"x1": Xin['selfspeedmag'],"nbases":params['nbases']}, return_type="dataframe")
        basis_x2 = patsy.dmatrix("cr(x2, df=nbases) - 1", {"x2": Xin['reldist'],"nbases":params['nbases']}, return_type="dataframe")
        basis_x3 = patsy.dmatrix("cr(x3, df=nbases) - 1", {"x3": Xin['wt'],"nbases":params['nbases']}, return_type="dataframe")
    elif params['basismain']=='bs':
        basis_x1 = patsy.dmatrix("bs(x1, degree=3,df=nbases) - 1", {"x1": Xin['selfspeedmag'], "nbases": params['nbases']},
                                 return_type="dataframe")
        basis_x2 = patsy.dmatrix("bs(x2, degree=3,df=nbases) - 1", {"x2": Xin['reldist'], "nbases": params['nbases']},
                                 return_type="dataframe")
        basis_x3 = patsy.dmatrix("bs(x3,degree=3, df=nbases) - 1", {"x3": Xin['wt'], "nbases": params['nbases']},
                                 return_type="dataframe")

    #Only for reward varaible
    if params['basistypeval'] == 'cr':
        basis_x4 = patsy.dmatrix("cr(x4, df=nbases) - 1", {"x4": Xin['relvalue'],"nbases":params['nbasis']}, return_type="dataframe")
    elif params['basistypeval'] == 'linear':
        basis_x4 = patsy.dmatrix("x4- 1", {"x4": Xin['relvalue']}, return_type="dataframe")
    elif params['basistypeval'] == 'bs0':
        '''
        Same as binning by creating piecewise constant basis functions
        '''
        knots = np.linspace(Xin['relvalue'].min(), Xin['relvalue'].max(), len(Xin['relvalue'].unique())+1)
        # Create the piecewise constant basis functions with the custom knots
        basis_x4= patsy.dmatrix("bs(x4, degree=0, knots=knots) - 1",
                                 {"x4": Xin['relvalue'], "knots": knots},
                                 return_type="dataframe")



    # List of basis matrices and penalty matrices for each variable
    basis_x_list = [jnp.array(basis_x1.values), jnp.array(basis_x2.values), jnp.array(basis_x3.values),jnp.array(basis_x4.values)]


    # Construct second-order difference matrices (D) and penalty matrices (S)
    S_list = []

    S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x1, basis_x2=None, is_tensor=False)))
    S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x2, basis_x2=None, is_tensor=False)))
    S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x3, basis_x2=None, is_tensor=False)))
    S_list.append(jnp.array(ut.smoothing_penalty_matrix(basis_x4, basis_x2=None, is_tensor=False)))

    beta_x_names = ['beta_x1', 'beta_x2', 'beta_x3', 'beta_x4']

    # Construct cont x cont interactions
    #1. reward and each variable
    if params['basismain']=='cr':
        if params['basistypeval'] == 'cr':
            tensor_basisx1x4 = patsy.dmatrix("te(cr(x1, df=nbases) ,cr(x4, df=nbasis) ) - 1",
                                         {"x1": Xin['selfspeedmag'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                          "nbasis": params['nbasis']}, return_type="dataframe")

            tensor_basisx2x4=patsy.dmatrix("te(cr(x2, df=nbases) ,cr(x4, df=nbasis) ) - 1",
                                         {"x2": Xin['reldist'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                          "nbasis": params['nbasis']}, return_type="dataframe")

            tensor_basisx3x4=patsy.dmatrix("te(cr(x3, df=nbases) ,cr(x4, df=nbasis) ) - 1",
                                         {"x3": Xin['wt'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                          "nbasis": params['nbasis']}, return_type="dataframe")
        elif params['basistypeval'] == 'linear':
            tensor_basisx1x4 = patsy.dmatrix("te(cr(x1, df=nbases), x4) - 1",
                                            {"x1": Xin['selfspeedmag'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                             }, return_type="dataframe")

            tensor_basisx2x4=patsy.dmatrix("te(cr(x2, df=nbases), x4) - 1",
                                            {"x2": Xin['reldist'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                             }, return_type="dataframe")

            tensor_basisx3x4=patsy.dmatrix("te(cr(x3, df=nbases), x4) - 1",
                                            {"x3": Xin['wt'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                             }, return_type="dataframe")
        elif params['basistype'] == 'bs0':
            knots = np.linspace(Xin['relvalue'].min(), Xin['relvalue'].max(), len(Xin['relvalue'].unique()) + 1)

            tensor_basisx1x4 = patsy.dmatrix("te(cr(x1, df=nbases),bs(x4, degree=0, knots=knots)) - 1",
                                             {"x1": Xin['selfspeedmag'],"x4":Xin['relvalue'],"nbases":params['nbases'], "knots": knots},
                                             return_type="dataframe")
            tensor_basisx2x4 =patsy.dmatrix("te(cr(x2, df=nbases),bs(x4, degree=0, knots=knots)) - 1",
                                             {"x2": Xin['reldist'],"x4":Xin['relvalue'],"nbases":params['nbases'], "knots": knots},
                                             return_type="dataframe")

            tensor_basisx3x4 =patsy.dmatrix("te(cr(x3, df=nbases),bs(x4, degree=0, knots=knots)) - 1",
                                             {"x3": Xin['wt'],"x4":Xin['relvalue'], "nbases":params['nbases'],"knots": knots},
                                             return_type="dataframe")
    if params['basismain']=='bs':
        if params['basistypeval'] == 'cr':
            tensor_basisx1x4 = patsy.dmatrix("te(bs(x1, degree=3,df=nbases) ,cr(x4, df=nbasis) ) - 1",
                                             {"x1": Xin['selfspeedmag'], "x4": Xin['relvalue'],
                                              "nbases": params['nbases'],
                                              "nbasis": params['nbasis']}, return_type="dataframe")

            tensor_basisx2x4 = patsy.dmatrix("te(bs(x2,degree=3, df=nbases) ,cr(x4, df=nbasis) ) - 1",
                                             {"x2": Xin['reldist'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                              "nbasis": params['nbasis']}, return_type="dataframe")

            tensor_basisx3x4 = patsy.dmatrix("te(bs(x3, degree=3,df=nbases) ,cr(x4, df=nbasis) ) - 1",
                                             {"x3": Xin['wt'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                              "nbasis": params['nbasis']}, return_type="dataframe")
        elif params['basistypeval'] == 'linear':
            tensor_basisx1x4 = patsy.dmatrix("te(bs(x1, degree=3,df=nbases), x4) - 1",
                                             {"x1": Xin['selfspeedmag'], "x4": Xin['relvalue'],
                                              "nbases": params['nbases'],
                                              }, return_type="dataframe")

            tensor_basisx2x4 = patsy.dmatrix("te(bs(x2, degree=3,df=nbases), x4) - 1",
                                             {"x2": Xin['reldist'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                              }, return_type="dataframe")

            tensor_basisx3x4 = patsy.dmatrix("te(bs(x3,degree=3, df=nbases), x4) - 1",
                                             {"x3": Xin['wt'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                              }, return_type="dataframe")
        elif params['basistype'] == 'bs0':
            knots = np.linspace(Xin['relvalue'].min(), Xin['relvalue'].max(), len(Xin['relvalue'].unique()) + 1)

            tensor_basisx1x4 = patsy.dmatrix("te(bs(x1, degree=3, df=nbases),bs(x4, degree=0, knots=knots)) - 1",
                                             {"x1": Xin['selfspeedmag'], "x4": Xin['relvalue'],
                                              "nbases": params['nbases'], "knots": knots},
                                             return_type="dataframe")
            tensor_basisx2x4 = patsy.dmatrix("te(bs(x2, degree=3,df=nbases),bs(x4, degree=0, knots=knots)) - 1",
                                             {"x2": Xin['reldist'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                              "knots": knots},
                                             return_type="dataframe")

            tensor_basisx3x4 = patsy.dmatrix("te(bs(x3,degree=3, df=nbases),bs(x4, degree=0, knots=knots)) - 1",
                                             {"x3": Xin['wt'], "x4": Xin['relvalue'], "nbases": params['nbases'],
                                              "knots": knots},
                                             return_type="dataframe")


    if params['basismain']=='cr':
        tensor_cont_basis = patsy.dmatrix("te(cr(x2, df=nbases) ,cr(x3, df=nbases) ) - 1",
                                 {"x2":Xin['reldist'],"x3": Xin['wt'],"nbases":params['nbases']}, return_type="dataframe")
    elif params['basismain']=='bs':
        tensor_cont_basis = patsy.dmatrix("te(bs(x2, degree=3,df=nbases) ,bs(x3,degree=3, df=nbases) ) - 1",
                                          {"x2": Xin['reldist'], "x3": Xin['wt'], "nbases": params['nbases']},
                                          return_type="dataframe")

    tensor_basis=[jnp.array(tensor_cont_basis.values),
              jnp.array(tensor_basisx1x4),
              jnp.array(tensor_basisx2x4),
              jnp.array(tensor_basisx3x4)]

    tx2x3 = ut.smoothing_penalty_matrix(basis_x2,basis_x3,is_tensor=True)[0]
    tx1x4 = ut.smoothing_penalty_matrix(basis_x1,basis_x4,is_tensor=True)[0]
    tx2x4 = ut.smoothing_penalty_matrix(basis_x2,basis_x4,is_tensor=True)[0]
    tx3x4 = ut.smoothing_penalty_matrix(basis_x3,basis_x4,is_tensor=True)[0]
    tensor_S = [tx2x3, tx1x4, tx2x4, tx3x4]

    return basis_x_list, S_list, tensor_basis, tensor_S, beta_x_names
