import torch

# name과 scheduler class를 묶는 entrypoint
_scheduler_entrypoints = {
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
}


# scheduler 이름을 가지는 class return
def scheduler_entrypoint(scheduler_name):
    return _scheduler_entrypoints[scheduler_name]


# scheduler list에 사용자가 전달한 이름이 존재하는지 판단
def is_scheduler(scheduler_name):
    return scheduler_name in _scheduler_entrypoints


def create_scheduler(scheduler_name, **kwargs):
    if is_scheduler(scheduler_name): # 정의한 scheduler list내에 이름이 있으면
        create_fn = scheduler_entrypoint(scheduler_name) # class를 가져와서
        scheduler = create_fn(**kwargs) # 전달된 인자를 사용해 선언
        
    else:
        raise RuntimeError('Unknown scheduler (%s)' % scheduler_name) # scheduler 없을 시, Error raise
        
    return scheduler # 정의된 scheduler return