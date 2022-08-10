/// <binding Clean='webpack' />
"use strict";

const gulp = require("gulp");
const webpack = require('webpack');
const webpackConfig = require('./webpack.config.js');
const gutil = require('gutil');

gulp.task("webpack", function (callback) {
    // run webpack
    webpack(webpackConfig, function (err, stats) {
        if (err) throw new gutil.PluginError("webpack", err);
        gutil.log("[webpack]", stats.toString({
            // output options
        }));
        callback();
    });
});