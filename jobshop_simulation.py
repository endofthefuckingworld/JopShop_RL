import simpy
import numpy as np
import pandas as pd

data_name = "ft06"
file_path = "Data/"+data_name+".xlsx"
process_t = pd.read_excel(file_path,sheet_name="process_time", header=None).to_numpy(dtype=np.int64)
machine_seq = pd.read_excel(file_path,sheet_name="machine_sequence", header=None).to_numpy(dtype=np.int64)

n_job = process_t.shape[0]
n_machine = process_t.shape[1]
inter_arrival = np.arange(n_job)
optimal = 55


class Order:
    def __init__(self, ID, j_type, arrival_time, machine_seq, process_time):
        self.id = ID
        self.type = int(j_type)
        self.arrival_t = arrival_time
        self.process_t = process_time
        self.machine_seq = machine_seq
        self.operation = 0

class Source:
    def __init__(self, name, factory):
        self.name = name
        self.fac = factory
        self.env = factory.env
        self.output = 0 #output: the number of jobs arrival
        
    def set_port(self):
        self.queue = self.fac.queue
        self.process = self.env.process(self.generate_order())
             
    def generate_order(self):
        while True:
            inter_arrival_t = inter_arrival[self.output]  #取得來到間隔時間
            yield self.env.timeout(inter_arrival_t)  #預約下一個事件
            order_type = self.output+1   #source output + 1
            order = Order(self.output, order_type, self.env.now, \
                machine_seq[self.output], process_t[self.output])  #建立工單物件
            print("{} : order {} ,type{} arrive".format(self.env.now, order.id, order.type))
            queue_id = order.machine_seq[order.operation]  #取得該工件第一站之機台
            self.queue[queue_id].order_arrival(order)  #工件來到第一站的等候區
            self.queue[queue_id].send_to_port()  #等候區選擇工件進行加工
            self.on_exit()

            if self.output >= n_job:  #如果所有工件皆已來到，停止來到流程
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
        if self.processor.status == True:  #如果機台空閒
            if len(self.space) == 1:  #如果等候區容量 = 1
                self.release_order()  #該工件立即加工
            elif len(self.space) > 1: #如果等候區容量 > 1
                self.fac.dispatcher.dispatch_for(self.id) #呼叫dispatcher進行派工(派工事件)
    
    def release_order(self, action = -1):
        self.sort_queue(action)  #依據派工法則排列等候區
        order = self.space[0]
        self.on_exit(order)  
        del self.space[0] #該派工法則權重最高的工件離開等候區
        self.processor.receive_order(order) #開始加工事件
    
    def sort_queue(self, rule_for_sorting):
        if rule_for_sorting == 0:  #FIFO
            self.space.sort(key = lambda entity : (entity.entry_queue_t, entity.id))

        elif rule_for_sorting == 1: #SPT
            self.space.sort(key = lambda entity : (entity.process_t[entity.operation], entity.id))
        
        elif rule_for_sorting == 2: #LPT
            self.space.sort(key = lambda entity : (-entity.process_t[entity.operation], entity.id))
        
    def order_arrival(self, order):
        self.on_entry(order)
        self.space.append(order)  #等候區新增該工件
    
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
        self.status = False  #宣告機台狀態為忙碌
        self.on_entry()
        print("{} : order {} ,type{} start treating at processor{}".format(self.env.now, order.id, order.type, self.id))
        self.env.process(self.process_order(order)) #執行加工流程
        
    def process_order(self, order):
        self.process_t = order.process_t[order.operation] #取得該operation加工時間
        yield self.env.timeout(self.process_t) #預排加工完成事件
        print("{} : order {} ,type{} finish treating at processor{}".format(self.env.now, order.id, order.type, self.id))  
        self.status = True #宣告機台狀態為空閒 
        order.operation += 1 #更新工件加工進度
        self.on_exit(order)
        
        if order.operation == len(order.machine_seq): #如果該工件已經加工完成
            self.sink.complete_order(order) #離開等候區
        else:
            queue_id = order.machine_seq[order.operation]  #取得下一站機台
            self.queue[queue_id].order_arrival(order) #進入下一站等候區
            self.queue[queue_id].send_to_port() #下一站等候區選擇工件進行加工
            
        self.in_queue.send_to_port() #當前機台等候區選擇工件進行加工
    
    def on_entry(self):
        self.utilization.change(self.env.now, 1)
        
    def on_exit(self, order):
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
            self.fac.terminal = True #模擬終止條件達成

        self.warehouse.append(order)
    
    def on_entry(self):
        self.fac.L.change(self.env.now, -1)
        
        
class Dispatcher:
    def __init__(self, factory):
        self.env = factory.env
        self.fac = factory
        self.dp_rule = 1  #SPT

    def dispatch_for(self, processor_id):
        #依據該派工法則指示等候區應釋出哪一個工件
        self.fac.queue[processor_id].release_order(self.dp_rule) 


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
        self.env = simpy.Environment()  #建立物件
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
        
        # next event algo --> simpy.env.step
        self.next_event = self.env.step  #simpy Environment method:執行下次事件
        self.terminal   = False #模擬終止條件起初設定為false，達成後變更為true
        
        # statistics
        self.L = L_calculator()
        self.W = W_calculator()
    
    

if __name__ == "__main__":

    jobshop_system = Factory()
    jobshop_system.build()
    while jobshop_system.terminal == False: #如果模擬終止條件未達成
        jobshop_system.next_event() #執行下次事件邏輯，時間推進至下次事件
