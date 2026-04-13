/**
 * API Service - Communicates with FastAPI backend
 */
(function() {
    'use strict';

    angular.module('forecastApp')
        .factory('ApiService', ['$http', function($http) {
            var BASE = '/api';

            return {
                getHealth: function() {
                    return $http.get(BASE + '/health');
                },
                getResults: function(sliceType) {
                    return $http.get(BASE + '/results/' + sliceType);
                },
                getChartData: function(sliceType) {
                    return $http.get(BASE + '/chart-data/' + sliceType);
                },
                getRadar: function(sliceType) {
                    return $http.get(BASE + '/radar/' + sliceType);
                },
                getPlots: function(sliceType) {
                    return $http.get(BASE + '/plots/' + sliceType);
                },
                getPredictions: function(sliceType) {
                    return $http.get(BASE + '/predictions?slice=' + sliceType);
                },
                getForecastSamples: function(slice) {
                    return $http.get(BASE + '/forecast/samples?slice=' + slice);
                },
                runForecast: function(params) {
                    return $http.post(BASE + '/forecast/run', params);
                }
            };
        }]);
})();
