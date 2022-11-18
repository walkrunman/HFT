from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

from simulator import MdUpdate, Order, OwnTrade, Sim, update_best_positions


class AlternativeStoikovStrategy:
    '''
        This strategy places ask and bid order every `delay` nanoseconds.
        If the order has not been executed within `hold_time` nanoseconds, it is canceled.
    '''
    def __init__(self, delay: float, hold_time:Optional[float] = None) -> None:
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
        
        # constant
        self.T = 10
        self.t = 0
        # mid price
        self.s = np.nan
        # quantity of assets
        self.q = 0
        
        self.gamma = 0.1
        self.k = 1.5
        # self.sigma2 = 6.25
        self.sigma2 = 0.6249 #std^2 per sec


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
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
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    
                    self.s = (best_bid + best_ask)/2
                    
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        if update.side == "ASK":
                            self.q = self.q + 0.001
                        else:
                            self.q = self.q - 0.001
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                
                r = self.s - self.q*self.gamma*self.sigma2*(self.T)
                spread = self.gamma*self.sigma2*(self.T) + \
                    (2/self.gamma)*np.log(1+self.gamma/self.k)
                    
                bid_price = r-spread/2
                ask_price = r+spread/2
                    
                #place order
                bid_order = sim.place_order( receive_ts, 0.001, 'BID', bid_price )
                ask_order = sim.place_order( receive_ts, 0.001, 'ASK', ask_price )
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders
    
class StoikovStrategy:
    def __init__(self, delay: float, t: np.int32, T: np.int32, hold_time:Optional[float] = None) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time
        
        # self.max_pos = 0.1
        
        # constant
        self.t = t
        self.T = T
        # mid price
        self.s = np.nan
        # quantity of assets
        self.q = 0
        
        self.gamma = 0.1
        self.k = 2.5
        # self.sigma2 = 6.25
        self.sigma2 = 0.6249 #std^2 per sec


    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
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
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    
                    self.s = (best_bid + best_ask)/2
                    
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        if update.side == "ASK":
                            self.q = self.q + 0.001
                        else:
                            self.q = self.q - 0.001
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'

            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                self.t = receive_ts
                
                r = self.s - self.q*self.gamma*self.sigma2*(self.T - self.t)*1e-9
                spread = self.gamma*self.sigma2*(self.T - self.t)*1e-9 + \
                    (2/self.gamma)*np.log(1+self.gamma/self.k)
                    
                bid_price = r-spread/2
                ask_price = r+spread/2
                    
                #place order
                # if np.abs(self.q) < self.max_pos:
                bid_order = sim.place_order( receive_ts, 0.001, 'BID', bid_price )
                ask_order = sim.place_order( receive_ts, 0.001, 'ASK', ask_price )
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders
    
class GStrategy:
    def __init__(self, delay: float, hold_time:Optional[float]) -> None:
        '''
            Args:
                delay(float): delay between orders in nanoseconds
                hold_time(Optional[float]): holding time in nanoseconds
        '''
        self.delay = delay
        if hold_time is None:
            hold_time = max( delay * 5, pd.Timedelta(10, 's').delta )
        self.hold_time = hold_time

        self.size_order = 0.0174
        
        self.a = 10.64
        self.b = 0.0004066
        self.c = 3.246
        
        self.delta = 0.001

    def run(self, sim: Sim ) ->\
        Tuple[ List[OwnTrade], List[MdUpdate], List[ Union[OwnTrade, MdUpdate] ], List[Order] ]:
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
        md_list:List[MdUpdate] = []
        #executed trades list
        trades_list:List[OwnTrade] = []
        #all updates list
        updates_list = []
        #current best positions
        best_bid = -np.inf
        best_ask = np.inf

        #last order timestamp
        prev_time = -np.inf
        #orders that have not been executed/canceled yet
        ongoing_orders: Dict[int, Order] = {}
        all_orders = []
        
        # Algorithm
        spread = 0
        mid_price = 0
        inventory = 0 # position
        
        while True:
            #get update from simulator
            receive_ts, updates = sim.tick()
            if updates is None:
                break
            #save updates
            updates_list += updates
            for update in updates:
                #update best position
                if isinstance(update, MdUpdate):
                    best_bid, best_ask = update_best_positions(best_bid, best_ask, update)
                    
                    spread = best_ask - best_bid
                    mid_price = (best_ask + best_bid)/2
                    
                    md_list.append(update)
                elif isinstance(update, OwnTrade):
                    trades_list.append(update)
                    #delete executed trades from the dict
                    if update.order_id in ongoing_orders.keys():
                        if update.side == "ASK":
                            inventory = inventory + self.size_order
                        else:
                            inventory = inventory - self.size_order
                        ongoing_orders.pop(update.order_id)
                else: 
                    assert False, 'invalid type of update!'
                    
            if receive_ts - prev_time >= self.delay:
                prev_time = receive_ts
                
                decicion_by_pos = self.make_decision(mid_price, inventory)
                
                if decicion_by_pos == "sell at market":
                    ask_order = sim.place_order( receive_ts, self.size_order, 'ASK', best_ask )
                elif decicion_by_pos == "buy at market":
                    bid_order = sim.place_order( receive_ts, self.size_order, 'BID', best_bid )
                elif decicion_by_pos == "BaBb":
                    bid_order = sim.place_order( receive_ts, self.size_order, 'BID', best_bid )
                    ask_order = sim.place_order( receive_ts, self.size_order, 'ASK', best_ask )
                elif decicion_by_pos == "Ba-Bb":
                    bid_order = sim.place_order( receive_ts, self.size_order, 'BID', best_bid )
                    ask_order = sim.place_order( receive_ts, self.size_order, 'ASK', best_ask - self.delta )
                elif decicion_by_pos == "BaBb+":
                    bid_order = sim.place_order( receive_ts, self.size_order, 'BID', best_bid + self.delta )
                    ask_order = sim.place_order( receive_ts, self.size_order, 'ASK', best_ask )
                elif decicion_by_pos == "Ba-Bb+":
                    bid_order = sim.place_order( receive_ts, self.size_order, 'BID', best_bid + self.delta)
                    ask_order = sim.place_order( receive_ts, self.size_order, 'ASK', best_ask - self.delta)
                    
                
                # bid_order = sim.place_order( receive_ts, self.min_pos, 'BID', best_bid )
                # ask_order = sim.place_order( receive_ts, self.min_pos, 'ASK', best_ask )
                ongoing_orders[bid_order.order_id] = bid_order
                ongoing_orders[ask_order.order_id] = ask_order

                all_orders += [bid_order, ask_order]
            
            to_cancel = []
            for ID, order in ongoing_orders.items():
                if order.place_ts < receive_ts - self.hold_time:
                    sim.cancel_order( receive_ts, ID )
                    to_cancel.append(ID)
            for ID in to_cancel:
                ongoing_orders.pop(ID)
            
                
        return trades_list, md_list, updates_list, all_orders
    
    def right_log(self, S, Y):
        try:
            if np.isnan(np.sign(S - np.log(self.a*Y))):
                return 1
            else:
                return np.sign(S - np.log(self.a*Y))
        except Exception:
            return 1
    
    def left_log(self, S, Y):
        try:
            if np.isnan(np.sign(S - np.log(-self.a*Y))):
                return 1
            else:
                return np.sign(S - np.log(-self.a*Y))
        except Exception:
            return 1
    
    def right_line(self, S, Y):
        return S - self.b - self.c*Y
    
    def left_line(self, S, Y):
        return S - self.b + self.c*Y
    
    def make_decision(self, S, Y):
        if self.right_log(S, Y) < 0:
            return "sell at market"
        elif self.left_log(S, Y) < 0:
            return "buy at market"
        elif self.right_line(S, Y) < 0 and self.left_line(S, Y) < 0:
            return "BaBb"
        elif self.right_line(S, Y) > 0 and self.left_line(S, Y) > 0:
            return "Ba-Bb+"
        elif self.right_line(S, Y) < 0 and self.left_line(S, Y) > 0:
            return "Ba-Bb"
        elif self.right_line(S, Y) > 0 and self.left_line(S, Y) < 0:
            return "BaBb+"
        
if __name__ == "__main__":
    from simulator import Sim
    # from strategy import BestPosStrategy
    from get_info import get_pnl
    from load_data import load_md_from_file
    
    
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    
    PATH_TO_FILE = r'C:\Users\walkr\Documents\HFT\HFT\simulator\data'
    # NROWS = 1_000_000
    NROWS = 1_000
    
    md = load_md_from_file(path=PATH_TO_FILE, nrows=NROWS)
    latency = pd.Timedelta(10, 'ms').delta
    md_latency = pd.Timedelta(10, 'ms').delta
    
    sim = Sim(md, latency, md_latency)
    
    delay = pd.Timedelta(0.1, 's').delta

    hold_time = pd.Timedelta(10, 's').delta
    
    strategy = GStrategy(delay, hold_time)
    trades_list, md_list, updates_list, all_orders = strategy.run(sim)
    
    # S = 0
    # Y = 2
    # print(S-np.log(Y))
    
    df = get_pnl(updates_list)
    
    dt = pd.to_datetime(df.receive_ts)
    
    plt.figure(figsize=(10, 5))
    plt.plot(dt, df.total, '-')
    plt.xlabel("time", fontsize=13)
    plt.ylabel("PnL", fontsize=13)
    plt.title("BestStrategy PnL", fontsize=15)
    plt.grid()
    plt.show()

