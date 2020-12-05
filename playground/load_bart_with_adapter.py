from transformers import BartModel
from bart_with_adapter import BartWithAdapterConfig, MyBartWithAdapter

def main():
    config = BartWithAdapterConfig.from_pretrained('facebook/bart-base')
    bart = MyBartWithAdapter(config)

    bart_old = BartModel.from_pretrained("facebook/bart-base")
    ret = bart.model.load_state_dict(bart_old.state_dict(), strict=False)

    print(ret)

if __name__ == "__main__":
    main()