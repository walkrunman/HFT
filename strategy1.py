from typing import List, Optional, Tuple, Union, Dict

import numpy as np
from scipy.linalg import fractional_matrix_power as mpow
import pandas as pd

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class Strategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay, hold_time,
                 gamma, big_gamma, ksi, intensity_btc, intensity_eth,
                 big_delta_btc, big_delta_eth, A, k
                 ) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        
        # self.max_pos = 0.001*20
        

        # mid price
        self.s = np.nan
        # quantity of assets
        self.q_btc = 0
        self.q_eth = 0
        
        self.gamma = 1
        self.big_gamma = big_gamma
        
        self.ksi = ksi 
        self.intensity_btc = intensity_btc
        self.intensity_eth = intensity_eth
        
        self.big_delta_btc = big_delta_btc
        self.big_delta_eth = big_delta_eth
        
        self.A = A
        self.k = k

    def run(self, sim_btc: Sim, sim_eth: Sim ):
        '''
            This function runs simulation

            Args:
                sim(Sim): simulator
            Returns:
                trades_list(List[OwnTrade]): list of our executed trades
                md_list(List[MdUpdate]): list of market data received by strategy
                updates_list( List[ Union[OwnTrade, MdUpdate] ] ): list of all updates 
                received by strategy(market data and information about executed trades)
                all_orders(List[Orted]): list of all placed orders
        '''

        #market data list
        md_list_btc:List[MdUpdate] = []
        #executed trades list
        trades_list_btc:List[OwnTrade] = []
        #all updates list
        updates_list_btc = []
        #current best positions
        best_bid_btc = -np.inf
        best_ask_btc = np.inf

        #last order timestamp
        prev_time_btc = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders_btc: Dict[int, Order] = {}
        all_orders_btc = []
        
        
        #market data list
        md_list_eth:List[MdUpdate] = []
        #executed trades list
        trades_list_eth:List[OwnTrade] = []
        #all updates list
        updates_list_eth = []
        #current best positions
        best_bid_eth = -np.inf
        best_ask_eth = np.inf

        #last order timestamp
        prev_time_eth = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders_eth: Dict[int, Order] = {}
        all_orders_eth = []
        
        while True:
            #get update from simulator
            receive_ts_btc, updates_btc = sim_btc.tick()
            receive_ts_eth, updates_eth = sim_eth.tick()
            
            if receive_ts_btc < receive_ts_eth:
                if updates_btc is None:
                    break
                #save updates
                updates_list_btc += updates_btc
                for update_btc in updates_btc:
                    #update best position
                    if isinstance(update_btc, MdUpdate):
                        best_bid_btc, best_ask_btc = update_best_positions(best_bid_btc, best_ask_btc, update_btc)
                        
                        self.s_btc = (best_bid_btc + best_ask_btc)/2
                        
                        md_list_btc.append(update_btc)
                        
                    elif isinstance(update_btc, OwnTrade):
                        trades_list_btc.append(update_btc)
                        #delete executed trades from the dict
                        if update_btc.order_id in ongoing_orders_btc.keys():
                            if update_btc.side == "ASK":
                                self.q_btc = self.q_btc - 0.001
                            else:
                                self.q_btc = self.q_btc + 0.001
                            ongoing_orders_btc.pop(update_btc.order_id)
                    else: 
                        assert False, 'invalid type of update!'
                        
                    if receive_ts_btc - prev_time_btc >= self.delay:
                        prev_time_btc = receive_ts_btc
                            
                        bid_price = self.s_btc + delta_bid(0, self.q_btc, self.q_eth, self.gamma, 
                                  self.big_gamma, self.intensity_btc, 
                                  self.ksi, self.big_delta_btc, self.A, self.k)
                        ask_price = self.s_btc - delta_ask(0, self.q_btc, self.q_eth, self.gamma, 
                                  self.big_gamma, self.intensity_btc, 
                                  self.ksi, self.big_delta_btc, self.A, self.k)
                            
                        #place order
                        bid_order_btc = sim_btc.place_order( receive_ts_btc, 0.001, 'BID', bid_price )
                        ask_order_btc = sim_btc.place_order( receive_ts_btc, 0.001, 'ASK', ask_price )
                        ongoing_orders_btc[bid_order_btc.order_id] = bid_order_btc
                        ongoing_orders_btc[ask_order_btc.order_id] = ask_order_btc
        
                        all_orders_btc += [bid_order_btc, ask_order_btc]
                    
                    to_cancel_btc = []
                    for ID, order in ongoing_orders_btc.items():
                        if order.place_ts < receive_ts_btc - self.hold_time:
                            sim_btc.cancel_order( receive_ts_btc, ID )
                            to_cancel_btc.append(ID)
                    for ID in to_cancel_btc:
                        ongoing_orders_btc.pop(ID)
                        
                receive_ts_btc, updates_btc = sim_btc.tick()
            else:
                if updates_eth is None:
                    break
                #save updates
                updates_list_eth += updates_eth
                for update_eth in updates_eth:
                    #update best position
                    if isinstance(update_eth, MdUpdate):
                        best_bid_eth, best_ask_eth = update_best_positions(best_bid_eth, best_ask_eth, update_eth)
                        
                        self.s_eth = (best_bid_eth + best_ask_eth)/2
                        
                        md_list_eth.append(update_eth)
                        
                    elif isinstance(update_eth, OwnTrade):
                        trades_list_eth.append(update_eth)
                        #delete executed trades from the dict
                        if update_eth.order_id in ongoing_orders_eth.keys():
                            if update_eth.side == "ASK":
                                self.q_eth = self.q_eth - 0.001
                            else:
                                self.q_eth = self.q_eth + 0.001
                            ongoing_orders_eth.pop(update_eth.order_id)
                    else: 
                        assert False, 'invalid type of update!'
                        
                    if receive_ts_eth - prev_time_eth >= self.delay:
                        prev_time_eth = receive_ts_eth
                            
                        bid_price = self.s_eth + delta_bid(1, self.q_btc, self.q_eth, self.gamma, 
                                  self.big_gamma, self.intensity_btc, 
                                  self.ksi, self.big_delta_btc, self.A, self.k)
                        ask_price = self.s_eth - delta_ask(1, self.q_btc, self.q_eth, self.gamma, 
                                  self.big_gamma, self.intensity_btc, 
                                  self.ksi, self.big_delta_btc, self.A, self.k)
                            
                        #place order
                        bid_order_eth = sim_eth.place_order( receive_ts_eth, 0.001, 'BID', bid_price )
                        ask_order_eth = sim_eth.place_order( receive_ts_eth, 0.001, 'ASK', ask_price )
                        ongoing_orders_eth[bid_order_eth.order_id] = bid_order_eth
                        ongoing_orders_eth[ask_order_eth.order_id] = ask_order_eth
        
                        all_orders_eth += [bid_order_eth, ask_order_eth]
                    
                    to_cancel_eth = []
                    for ID, order in ongoing_orders_eth.items():
                        if order.place_ts < receive_ts_eth - self.hold_time:
                            sim_eth.cancel_order( receive_ts_eth, ID )
                            to_cancel_eth.append(ID)
                    for ID in to_cancel_eth:
                        ongoing_orders_eth.pop(ID)
                
                receive_ts_eth, updates_eth = sim_eth.tick()
            
                
        return trades_list_btc, md_list_btc, updates_list_btc, all_orders_btc, trades_list_eth, md_list_eth, updates_list_eth, all_orders_eth
    
def C_ksi(big_delta, k, ksi):
    if ksi > 0:
        C_ksi_ = (1+ksi*big_delta/k)**(-(k/(ksi*big_delta))-1)
        return C_ksi_
    elif k == 0:
        C_ksi_ = np.exp(-1)
        return C_ksi_
    else:
        raise ValueError('Invalid big_delta in function C_ksi_f')
        
def H_ksi(p, A, k, big_delta, ksi):
    H_ksi_ = (A*big_delta/k)*C_ksi(big_delta, k, ksi)*np.exp(-k*p)
    return H_ksi_

def H_ksi_first_derivative(p, A, k, big_delta, ksi):
    H_ksi_fd = -A*big_delta*C_ksi(big_delta, k, ksi)*np.exp(-k*p)
    return H_ksi_fd

def H_ksi_second_derivative(p, A, k, big_delta, ksi):
    H_ksi_sd = k*A*big_delta*C_ksi(big_delta, k, ksi)*np.exp(-k*p)
    return H_ksi_sd

def D_f(A1, k1, big_delta1, ksi1, A2, k2, big_delta2, ksi2):
    H11 = H_ksi_second_derivative(0, A1, k1, big_delta1, ksi1)
    H22 = H_ksi_second_derivative(0, A2, k2, big_delta2, ksi2)
    
    D = np.array([[H11, 0],
                  [0, H22]])
    return D

# count 1 time per simulation
def E_f(sigma_btc, sigma_eth, rho):
    E = np.array([[sigma_btc, rho],
                  [rho, sigma_eth]])
    return E

# count 1 time per simulation
def big_gamma_f(E, A1, k1, big_delta1, ksi1, A2, k2, big_delta2, ksi2):
    D = D_f(A1, k1, big_delta1, ksi1, A2, k2, big_delta2, ksi2)
    
    Dp12 = mpow(D, 1/2)
    Dm12 = mpow(D, -1/2)
    
    big_gamma = mpow(np.matmul(np.matmul(Dp12,E),Dp12), 1/2)
    big_gamma = np.matmul(Dm12, big_gamma)
    big_gamma = np.matmul(big_gamma, Dm12)
    
    return big_gamma

# each time
def delta_star(p, intensity, ksi, big_delta, A, k):
    delta = (intensity**(-1))*(ksi*H_ksi(p, A, k, big_delta, ksi) -\
                    H_ksi_first_derivative(p, A, k, big_delta, ksi)/big_delta)
    return delta

# btc_or_eth = 0 => btc, btc_or_eth = 1 => eth
def delta_bid(btc_or_eth, q1, q2, gamma, big_gamma, intensity, ksi, big_delta, A, k):
    q = q1 if btc_or_eth==0 else q2
    p = np.sqrt(gamma/2)*big_gamma[0][0]*(2*q + big_delta)/2 +\
        big_gamma[btc_or_eth][0]*q1 + big_gamma[btc_or_eth][1]*q2
    
    delta_bid_ = delta_star(p, intensity, ksi, big_delta, A, k)

    return delta_bid_

def delta_ask(btc_or_eth, q1, q2, gamma, big_gamma, intensity, ksi, big_delta, A, k):
    q = q1 if btc_or_eth==0 else q2
    p = -np.sqrt(gamma/2)*big_gamma[0][0]*(2*q - big_delta)/2 +\
        big_gamma[btc_or_eth][0]*q1 + big_gamma[btc_or_eth][1]*q2
    
    delta_ask_ = delta_star(p, intensity, ksi, big_delta, A, k)
    
    return delta_ask_

    
if __name__ == "__main__":
    from simulator import Sim
    # from strategy import BestPosStrategy
    from get_info import get_pnl
    from load_data import load_md_from_file
    
    
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    
    PATH_TO_BTC = r'C:\Users\walkr\Documents\HFT\HFT\simulator\data\BTC'
    PATH_TO_ETH = r'C:\Users\walkr\Documents\HFT\HFT\simulator\data\ETH'
    # NROWS = 1_000_000
    NROWS = 100_000
    
    md_btc = load_md_from_file(path=PATH_TO_BTC, nrows=NROWS)
    md_eth = load_md_from_file(path=PATH_TO_ETH, nrows=NROWS)
    latency = pd.Timedelta(10, 'ms').delta
    md_latency = pd.Timedelta(10, 'ms').delta
    
    sim_btc = Sim(md_btc, latency, md_latency)
    sim_eth = Sim(md_eth, latency, md_latency)
    
    delay = pd.Timedelta(0.1, 's').delta
    hold_time = pd.Timedelta(10, 's').delta
    
    
    
    sigma_btc = 1
    sigma_eth = 1
    rho = 0.4
    
    gamma = 0.4
    A1 = 1
    k1 = 1
    big_delta1 = 1
    ksi1 = 1
    A2 = 1
    k2 = 1
    big_delta2 = 1
    ksi2 = 1
    
    A2 = A1
    A = A2
    
    E = E_f(sigma_btc, sigma_eth, rho)
    big_gamma = big_gamma_f(E, A1, k1, big_delta1, ksi1, A2, k2, big_delta2, ksi2)
    
    ksi = 1
    
    intensity_btc = 1
    intensity_eth = 1
    
    big_delta_btc = 1
    big_delta_eth = 1
    A = 1
    k = 1
    
    strategy = Strategy(delay, hold_time,
                        gamma, big_gamma, ksi, intensity_btc, intensity_eth,
                        big_delta_btc, big_delta_eth, A, k
                        )
    trades_list_btc, md_list_btc, updates_list_btc, all_orders_btc, trades_list_eth, md_list_eth, updates_list_eth, all_orders_eth = strategy.run(sim_btc, sim_eth)
    
    
    df = get_pnl(updates_list_btc)
    
    dt = pd.to_datetime(df.receive_ts)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dt, df.total, '-')
    plt.xlabel("time", fontsize=13)
    plt.ylabel("PnL", fontsize=13)
    plt.title("BestStrategy PnL", fontsize=15)
    plt.grid()
    plt.show()
    
    df = get_pnl(updates_list_eth)
    
    dt = pd.to_datetime(df.receive_ts)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dt, df.total, '-')
    plt.xlabel("time", fontsize=13)
    plt.ylabel("PnL", fontsize=13)
    plt.title("BestStrategy PnL", fontsize=15)
    plt.grid()
    plt.show()
    
