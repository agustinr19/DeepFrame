def train(train_dataloader,model,criterion,optimizer,epochs):
	#TODO
	model.train() # switch to train mode
	for i, (inp, target) in enumerate(train_dataloader):
		pred = model(inp)
		loss = criterion(pred, target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def main():
	# TODO 
	pass

if __name__ == '__main__':
	main()