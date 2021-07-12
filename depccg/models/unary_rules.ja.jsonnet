{
  unary_rules: [
    ['S[mod=adn,form=imp,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adv,form=stem,fin=f]', 'S[mod=X1,form=X2,fin=X3]/S[mod=X1,form=X2,fin=X3]'],
    ['S[mod=adn,form=attr,fin=f]\\NP[case=ga,mod=nm,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adn,form=base,fin=f]\\NP[case=ga,mod=nm,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adv,form=cont,fin=f]\\NP[case=ga,mod=nm,fin=f]', '(S[mod=X1,form=X2,fin=X3]\\NP[case=ga,mod=nm,fin=f])/(S[mod=X1,form=X2,fin=X3]\\NP[case=ga,mod=nm,fin=f])'],
    ['S[mod=adn,form=attr,fin=f]\\NP[case=o,mod=nm,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adn,form=base,fin=f]\\NP[case=o,mod=nm,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['NP[case=nc,mod=adv,fin=f]', 'S[mod=X1,form=X2,fin=X3]/S[mod=X1,form=X2,fin=X3]'],
    ['S[mod=adn,form=base,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adv,form=cont,fin=f]', 'S[mod=X1,form=X2,fin=X3]/S[mod=X1,form=X2,fin=X3]'],
    ['S[mod=adn,form=hyp,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adn,form=stem,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adv,form=hyp,fin=f]', 'S[mod=X1,form=X2,fin=X3]/S[mod=X1,form=X2,fin=X3]'],
    ['S[mod=adn,form=cont,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
    ['S[mod=adn,form=attr,fin=f]', 'NP[case=nc,mod=X1,fin=X2]/NP[case=nc,mod=X1,fin=X2]'],
  ],
}
