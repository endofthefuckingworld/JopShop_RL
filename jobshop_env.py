import simpy
import numpy as np
import pandas as pd

data_name = "ft06"
file_path = "Data/"+data_name+".xlsx"
process_t = pd.read_excel(file_path,sheet_name="process_time", header=None).to_numpy(dtype=np.int64)
machine_seq = pd.read_excel(file_path,sheet_name="machine_sequence", header=None).to_numpy(dtype=np.int64)

n_job = process_t.shape[0]
n_machine = process_t.shape[1]
action_space = 8
inter_arrival = np.zeros(n_job)
optimal = 55
setup_t = np.zeros((n_job,n_job))

weights = []
for i in range(n_machine):
    weights.append(sum(process_t[np.where(machine_seq == i)]))

class Order:
    def __init__(self, ID, j_type, arrival_time, machine_seq, process_time):
        self.id = ID
        self.type = int(j_type)
        self.arrival_t = arrival_time
        self.process_t = process_time[np.where(process_time!=-1)]
        self.machine_seq = machine_seq[np.where(machine_seq!=-1)]
        self.operation = 0

class Source:
    def __init__(self, name, factory):
        self.name = name
        self.fac = factory
        self.env = factory.env
        #output: the number of jobs arrival
        self.output = 0
        
    def set_port(self):
        self.queue = self.fac.queue
        self.process = self.env.process(self.generate_order())
             
    def generate_order(self):
        queue_ids = []
        while True:
            inter_arrival_t = inter_arrival[self.output]
            yield self.env.timeout(inter_arrival_t)
            order_type = self.output+1
            order = Order(self.output, order_type, self.env.now, machine_seq[self.output], process_t[self.output])
            #print("{} : order {} ,type{} arrive".format(self.env.now, order.id, order.type))
            queue_id = order.machine_seq[order.operation]

            if queue_id not in queue_ids:
                queue_ids.append(queue_id)

            self.queue[queue_id].order_arrival(order)
            self.on_exit()
        
            if self.output < len(inter_arrival):
                if inter_arrival[self.output] == 0:
                    continue

            for i in queue_ids:
                self.queue[i].send_to_port()

            queue_ids.clear()

            if self.output >= n_job:
                break

    def on_exit(self):
        self.output += 1
        self.fac.L.change(self.env.now, 1)
        
        
class Queue:
    def __init__(self, factory, id, name):
        self.id = id
        self.name = name
        self.fac = factory
        self.env = factory.env
        self.space = []    #queue
        
    def set_port(self, output_port):
        self.processor = output_port
        
    def send_to_port(self):
        if self.processor.status == True:
            if len(self.space) == 1:
                self.release_order()
            elif len(self.space) > 1:
                self.fac.dispatcher.dispatch_for(self.id)
    
    def release_order(self, action = -1):
        self.sort_queue(action)
        order = self.space[0]
        self.on_exit(order)
        del self.space[0]
        self.processor.receive_order(order)
    
    def sort_queue(self, rule_for_sorting):
        if rule_for_sorting == 0:  #FIFO
            self.space.sort(key = lambda entity : (entity.entry_queue_t, entity.id))

        elif rule_for_sorting == 1: #SPT
            self.space.sort(key = lambda entity : (entity.process_t[entity.operation], entity.id))
        
        elif rule_for_sorting == 2: #LPT
            self.space.sort(key = lambda entity : (-entity.process_t[entity.operation], entity.id))
        
        elif rule_for_sorting == 3: #LRM
            self.space.sort(key = lambda entity : (-entity.lrm, entity.id))
        
        elif rule_for_sorting == 4: #LSO
            self.space.sort(key = lambda entity : (-entity.so, entity.id))

        elif rule_for_sorting == 5: #SSO
            self.space.sort(key = lambda entity : (entity.so, entity.id))
        
        elif rule_for_sorting == 6: #SRM
            self.space.sort(key = lambda entity : (sum(entity.process_t[entity.operation:]), entity.id))
        
        elif rule_for_sorting == 7: #LRM
            self.space.sort(key = lambda entity : (-sum(entity.process_t[entity.operation:]), entity.id))
        
    def order_arrival(self, order):
        self.on_entry(order)
        self.space.append(order)
    
    def on_entry(self, order):
        order.entry_queue_t = self.env.now
        
    def on_exit(self, order):
        order.exit_queue_t = self.env.now
        self.fac.W.calculate_mean(order.exit_queue_t - order.entry_queue_t)

class Processor:
    def __init__(self, factory, Processor_id, name):
        self.id = Processor_id
        self.name = name
        self.fac = factory
        self.status = True    #processor is free or not
        self.env = factory.env
        self.type_now = 0
        self.previous_type = 0
        self.utilization = L_calculator()

    def set_port(self, queue):
        self.in_queue = queue
        self.queue = self.fac.queue
        self.sink = self.fac.sink
        
    def receive_order(self, order):
        self.status = False
        self.on_entry()
        #print("{} : order {} ,type{} start treating at processor{}".format(self.env.now, order.id, order.type, self.id))
        self.type_now = order.type
        self.setup_t = setup_t[self.previous_type - 1][order.type - 1] if self.previous_type != 0 else 0
        self.process_t = order.process_t[order.operation]
        self.operation_now = (order.id, order.operation)
        self.available_t = self.env.now + self.setup_t + self.process_t
        self.env.process(self.setup_processor(order))
    
    def setup_processor(self, order):
        yield self.env.timeout(self.setup_t)
        self.env.process(self.process_order(order))
        
    def process_order(self, order):
        
        yield self.env.timeout(self.process_t)
        #print("{} : order {} ,type{} finish treating at processor{}".format(self.env.now, order.id, order.type, self.id))   
        self.on_exit(order)
        order.operation += 1
        self.status = True
        
        if order.operation == len(order.machine_seq):
            self.sink.complete_order(order)
        else:
            queue_id = order.machine_seq[order.operation]
            self.queue[queue_id].order_arrival(order)
            self.queue[queue_id].send_to_port()
            
        self.previous_type = order.type
        self.in_queue.send_to_port()
    
    def on_entry(self):
        self.utilization.change(self.env.now, 1)
        
    def on_exit(self, order):
        self.fac.operation_state[order.id, order.operation] = 1
        self.utilization.change(self.env.now, -1)


class Sink:
    def __init__(self, factory):
        self.env = factory.env
        self.input = 0
        self.warehouse = []
        self.fac = factory
          
    def complete_order(self, order):
        self.input += 1 
        self.on_entry()
        
        if self.input >= n_job: #如果所有工件皆加工完成
            self.fac.learning_event = True 
            self.fac.terminal.succeed()

        self.warehouse.append(order)
    
    def on_entry(self):
        self.fac.L.change(self.env.now, -1)
        
        
class Dispatcher:
    def __init__(self, factory):
        self.env = factory.env
        self.fac = factory
        self.listeners = []

    def dispatch_for(self, processor_id):
        self.fac.learning_event = True  #宣告learning event為true
        self.listeners.append(self.fac.processor[processor_id]) #將需要派工之機台存入listener，等候派工決策

    def execute_decision(self, action):
        for listener in self.listeners:
            listener.queue[listener.id].release_order(action) #依據派工決策為所有待派工之機台派工
        
        self.listeners = [] #派工完畢，清空listener


#Statistics
class L_calculator:
    def __init__(self):
        self.cumulative = 0
        self.time_lower = 0
        self.time_upper = 0
        self.L_now = 0
        self.L = 0
        
    def change(self, time_now, change):
        self.time_upper = time_now
        self.cumulative += (self.time_upper - self.time_lower) * self.L_now
        self.L_now += change
        self.time_lower = time_now

    def reset(self,time_now, L_now):
        self.cumulative = 0
        self.time_upper = 0
        self.L = 0
        self.time_lower = time_now
        self.L_now = L_now

    def caculate_mean(self, time_now):
        self.change(time_now, 0)
        self.L = self.cumulative / (self.time_upper+1e-8)
        return self.L

class W_calculator:
    def __init__(self):
        self.output = 0
        self.mean_waiting_time = 0
    
    def calculate_mean(self, waiting_t):
        self.output += 1
        self.mean_waiting_time += (waiting_t - self.mean_waiting_time)/self.output
        

class Factory:   
    def build(self):  
        self.env = simpy.Environment()
        self.n_processor = n_machine
        self.queue = []
        self.processor = []
        self.source = Source('source', self)
        self.sink = Sink(self)
        self.dispatcher = Dispatcher(self)

        for i in range(self.n_processor):
            queue = Queue(self, i, 'queue_'+str(i))
            processor = Processor(self, i, 'processor_'+str(i))
            self.queue.append(queue)
            self.processor.append(processor)
            self.queue[i].set_port(self.processor[i])
            self.processor[i].set_port(self.queue[i])
 
        self.source.set_port()
        
        self.learning_event = False #判斷是否為學習事件，一開始為false
        
        # next event algo --> simpy.env.step
        self.next_event = self.env.step
        self.terminal   = self.env.event()
        
        # statistics
        self.L = L_calculator()
        self.W = W_calculator()

        # initial_state
        self.operation_state = np.zeros((machine_seq.shape[0],machine_seq.shape[1]))

        self.uti = np.zeros(self.n_processor)
    
    def reset(self):
        self.build()  #建立物件
        while not self.learning_event: #不斷進行事件推進直到學習事件
            self.next_event() #執行下次事件邏輯，時間推進至下次事件
        self.learning_event = False #將學習事件重置為false
        state = self.get_state() #蒐集狀態
        return state
    
    def step(self, action):
        self.learning_event = False #將學習事件重置為false
        self._pass_action(action) #檢測動作是否合法

        while not self.learning_event: #不斷進行事件推進直到學習事件
            self.next_event() #執行下次事件邏輯，時間推進至下次事件

        state = self.get_state() #蒐集狀態
        done = self.terminal.triggered #取得done:代表是否結束
        reward = self.compute_reward(done) #計算獎勵
        inf = None
        if done:
            inf = self.env.now

        return state, reward, done, inf
    
    def _pass_action(self, action):
        ## execute the action ##
        assert action in np.arange(action_space)
        self.dispatcher.execute_decision(action)
    
    def get_state(self):
        state = np.zeros((5,machine_seq.shape[0],machine_seq.shape[1]))
        #p_utilization = np.zeros(n_machine)

        for m in self.dispatcher.listeners:
            for j in self.queue[m.id].space:
                state[0, j.id, j.operation] = 1
        
        for m in self.processor:
            if m.status == False:
                #p_utilization[m.id] = m.utilization.caculate_mean(self.env.now)
                i,j = m.operation_now
                mat = m.available_t
                state[1,i,j] = mat
        """
        for i in range(machine_seq.shape[0]):
            for j in range(machine_seq.shape[1]):
                state[2,i,j] = p_utilization[machine_seq[i,j]]
        """
        state[2] = self.operation_state
        state[3] = machine_seq
        state[4] = process_t
        
        return state
    
    def compute_reward(self, done):
        reward = 0
        uti_weight = np.ones(self.n_processor)/self.n_processor
        gamma = 1200

        for m in self.processor:
            uti_now = m.utilization.caculate_mean(self.env.now)
            reward += uti_weight[m.id]*(uti_now - self.uti[m.id])
            self.uti[m.id] = uti_now
        
        if done:
            if  self.env.now == optimal:
                reward += gamma
            else:
                reward += gamma/abs(self.env.now-optimal)
        
        return reward