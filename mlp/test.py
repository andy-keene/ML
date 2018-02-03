
class gen(object):
    def __init__(self):
        self.batch_size = 5
        pass
    
    def minibatch(self, inputs, targets):
        start = 0
        while start < inputs.shape[0]:
            yield inputs[start:start + self.batch_size], targets[start:start+self.batch_size]
            start += self.batch_size