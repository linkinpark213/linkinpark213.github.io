photo_waterfall = {
    page: 1,
    offset: 20,
    folder: '',
    init: function (folder) {
        var that = this;
        that.folder = folder;
        $.getJSON(folder + "output.json", function (data) {
            that.render(that.page, data);
        });
    },

    render: function (page, data) {
        var begin = (page - 1) * this.offset;
        var end = page * this.offset;
        if (begin >= data.length) return;
        var html = "", li = "";
        var count = 0;
        for (var i = begin; i < end && i < data.length; i++) {
            var filename = data[i].filename;
            var description = data[i].description;

            li = '<a data-fancybox="gallery" href="' + this.folder + filename + '" class="card" pic-num="' + count + '">' +
                '<div class="ImageInCard">' +
                '<div>' +
                '<img src="' + this.folder + "thumbnail/" + filename + '" onload="photo_waterfall.minigrid();"/>' +
                '</div>' +
                '</div>' +
                '<div class="TextInCard">' + description + '</div>' +
                '</a>';
            html += li;
            count++;

        }

        $(".ImageGrid").append(html);
        this.minigrid();
    },

    minigrid: function () {
        var grid = new Minigrid({
            container: '.ImageGrid',
            item: '.card',
            gutter: 12
        });
        grid.mount();
        $(window).resize(function () {
            grid.mount();
        });
    }

}