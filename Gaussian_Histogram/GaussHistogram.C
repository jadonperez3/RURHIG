void myMacro() {
    TH1F *h = new TH1F("h", "My Histogram; X axis; Counts", 100, -5, 5);
    h->FillRandom("gaus", 10000);
    h->Draw();
}
