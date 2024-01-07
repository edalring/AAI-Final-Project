import os
from dataloader import MNISTDataset
import options
from utils import torch_utils
import torch


class Tester:
    def __init__(self,args, model, testloader):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.testloader = testloader
        

    def test(self):
        self.model.eval()
        result = []
        for data in self.testloader:
            preds, name = self.inference(data)
            print('preds.shape:', preds.shape)
            print('name.szie:', len(name))
            for i, pred in enumerate(preds):
                result.append((str(name[i]), str(int(torch.argmax(pred)))))
                # result.append((str(name[i]), str(pred)))

        result.sort(key= lambda x: int(x[0].rstrip('.npy')))
        path = 'output/result/'+str(self.args.model_type)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path+'/test.txt', 'w') as output_file:
            for item in result:
                output_file.write(item[0] +' '+ item[1]+'\n')


    def inference(self, data):
        img, name = data
        img = img.to(self.device)
        pred = self.model(img).to(self.device)
        return pred, name

def main():
    args = options.prepare_test_args()
    model = torch_utils.prepare_model(args, is_train=False)

    test_dataset = MNISTDataset(data_path=args.data_path, mode='test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True, pin_memory=True)
    tester = Tester(args, model, testloader)

    tester.test()

if __name__ == '__main__':
    main()