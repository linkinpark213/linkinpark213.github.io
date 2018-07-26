cards = [];
current = 0;
max = 0;
time_period = 3000;
function next(current) {
    var temp = parseInt(current);
    temp = temp >= max ? 0 : temp + 1;
    return temp;
}

function showText(current) {
    cards[current].children[1].style.opacity = 0.8;
}

function hideText(current) {
    cards[current].children[1].style.opacity = 0;
}

function showNext(current) {
    hideText(current);
    current = next(current);
    showText(current);
    return current;
}

window.onload = function() {
    cards = $('.card');
    showText(current);
    current = 0;
    max = cards.length - 1;
    photo_waterfall.minigrid();
    interval = window.setInterval(function(){
        current = showNext(current);
    }, time_period);
    $('.card').mouseenter(function() {
        var temp = this.getAttribute('pic-num');
        clearInterval(interval);
        hideText(current);
        current = temp;
        showText(current);
    });
    $('.card').mouseenter(function() {
        var temp = this.getAttribute('pic-num');
        clearInterval(interval);
        if(current != temp)
            hideText(current);
            current = temp;
            showText(current);
    });
    $('.card').mouseleave(function(){
        var temp = parseInt(this.getAttribute('pic-num'));
        interval = setInterval(function(){
            current = showNext(current);
        }, time_period)
    });
};
