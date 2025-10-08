from models.market_trend_model import MarketTrendModel

if __name__ == "__main__":
    model = MarketTrendModel()
    model.fit_synthetic(length=72)
    model.save()
    print("Saved:", model.model_path)


