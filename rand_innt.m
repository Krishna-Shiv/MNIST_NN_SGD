function w = rand_init(flayer_num,player_num)

    %random initialization 
    
    eps = sqrt(6/(flayer_num + flayer_num)); %ideal epsilon. very small
    
    w = rand(flayer_num, player_num+1)*2*eps - eps; %approx. gradient 
    
end