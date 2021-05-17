import React, {useEffect} from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ActivityIndicator, FlatList, Dimensions } from 'react-native';
import axios from 'axios';
import deviceStorage from '../Services/DeviceStorage';
import AsyncStorage  from '@react-native-async-storage/async-storage';
import Icons from 'react-native-vector-icons/MaterialIcons';
import TabBar from 'react-native-nav-tabbar';
import {ChonseSelect} from 'react-native-chonse-select';
import SegmentedControlTab from 'react-native-segmented-control-tab';
//import ReactWordcloud from 'react-wordcloud';

let url='http://127.0.0.1:5000';
// let url='http://192.168.0.146:5000';
const { width } = Dimensions.get('window');
const styles = StyleSheet.create({
  container: {
    flex: 1,
    flexDirection: 'column',
    //alignItems: 'center',
    //justifyContent: 'center',
    backgroundColor: '#D5F5E3',
    //marginTop : 8
  },
  tabContainerStyle:{
    flex: 1,
    paddingTop : 8
  },
  listStyle:{
    flex: 15,
    alignItems: 'center',
    justifyContent: 'flex-start',

  },
  loginBtn:{
    width:width-100,
    backgroundColor:"#fb5b5a",
    borderRadius:25,
    height:50,
    alignItems:"center",
    justifyContent:"center",
    marginTop:10,
    marginBottom:10,
  },
  White:{
    fontSize: 19,
    color : '#ffffff'
  },
  Orange:{
    fontSize : 20,
    color : "#ca5010"
  },
  horizontal: {
    flexDirection: "row",
    justifyContent: "space-around",
    padding: 10,
    color : '#DD2C00'
  },
  flatList:{
    justifyContent: 'center',
    flex: 1,
    marginLeft: 10,
    marginRight: 10,
    marginBottom: 10,
    marginTop: 10,
  },
  item: {
    width: width-10,
    alignItems:"center",
    marginVertical: 9,
  },
  wordStyle:{
    fontSize: 20,
    color : '#E67E22'
  },
  buttons:{
    alignItems: 'center',
    justifyContent: 'center',
  },
  loader:{
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  }
});

const HomeScreen = ({navigation}) => {
  const[topWords, setTopWords] = React.useState('');
  const[isTrue, setIsTrue] = React.useState(false);
  const[wordList, setWordList] = React.useState('');
  const[tweets, setTweets] = React.useState('');
  const[isLoading, setIsLoading] = React.useState(true);
  const[user, setUser] = React.useState('');
  const[index, setIndex] = React.useState(0);
  let word_link = undefined;

  useEffect(()=>{
    if(word_link == undefined){
      AsyncStorage.getItem('user_data', (err, result) => {
        if(result){
          // console.log('in load user', result)
          let userTemp =  JSON.parse(result);
          setUser(userTemp);
          // console.log("in news call ", user)
          // console.log(zip_code)
            let zip_code = userTemp['zip_code'];
            setIsLoading(true);
            let finalUrl = '';
            let finalTweetUrl = '';
            if(zip_code && zip_code!=undefined && zip_code!=''){
              finalUrl = url+"/news?zip="+zip_code;
              finalTweetUrl = url+"/getTweets";
              console.log(finalUrl);
              console.log(finalTweetUrl);
              axios.get(finalUrl, {
                headers: {
                  "content-type": "application/json",
                },
              })
              .then((response) => {
                console.log("news date",response.data.date);
                word_link = {}
                let l = [];
                response.data.data.forEach(element => {
                  console.log(element['word'])
                  l.push(element['word'])
                });
                setWordList(l);
                console.log(wordList);
                setTopWords(response.data.data);
                let payload_tweet = {
                  'zip' : zip_code,
                  'word' : l
                }
                if(l.length > 0){
                  axios.post(finalTweetUrl,payload_tweet, {
                  headers:{
                    "content-type":"application/json"
                }
                }).then((response)=>{
                    console.log("tweet date:",response.data.date);
                    setTweets(response.data);
                    setIsLoading(false);
                    console.log("isLoading: false");
                }).catch((err)=>{
                  console.log(err)
                });
                }
              })
              .catch((err) => {
                console.log(err);
              });
            }
        }
      });
    }
  },[]);
  const goBack = () => {
    setIsTrue(false);
  }
  const getNewsTest = (index = 0) =>{
      //alert(index);
      setIndex(index);
      index = index.toString();
      console.log(index == '0');
  }
  const getNews = (i = 0) =>{
    setIndex(i);
    indx = i.toString();
    let zip_code = user['zip_code'];
    setIsLoading(true);
    setIsTrue(true);
    let finalUrl = '';
    let finalTweetUrl = '';
    if(zip_code && zip_code!=undefined && zip_code!=''){
      finalUrl = url+"/news?zip="+zip_code;
      finalTweetUrl = url+"/getTweets";
      if(indx!='0'){
        console.log("not todays");
        finalUrl = '';
        finalTweetUrl = ''
        finalUrl = url+"/oldnews?zip="+zip_code+"&index="+indx;
        finalTweetUrl = url+"/oldtweets?zip="+zip_code+"&index="+indx;
      }
      console.log(finalUrl);
      console.log(finalTweetUrl);
      axios.get(finalUrl, {
        headers: {
          "content-type": "application/json",
        },
      })
      .then((response) => {
        console.log("news date",response.data.date);
        word_link = {}
        let l = [];
        response.data.data.forEach(element => {
          console.log(element['word'])
          l.push(element['word'])
        });
        setWordList(l);
        console.log(wordList);
        setTopWords(response.data.data);
        let payload_tweet = {
          'zip' : zip_code,
          'word' : l
        }
        if(l.length > 0){
          axios.post(finalTweetUrl,payload_tweet, {
          headers:{
            "content-type":"application/json"
        }
        }).then((response)=>{
            console.log("tweet date:",response.data.date);
            setTweets(response.data);
            setIsLoading(false);
            console.log("isLoading: false");
        }).catch((err)=>{
          console.log(err)
        });
        }
      })
      .catch((err) => {
        console.log(err);
      });
    }
  }

  return (
    <View style={styles.container}>
      <View style={styles.tabContainerStyle}>
      <SegmentedControlTab
        values={['Today', 'Yesterday', 'Day Before']}
        selectedIndex={index}
        onTabPress={getNews}
      />
      </View>
      <View style={styles.listStyle}>
      {isLoading?(<View style={styles.loader}><ActivityIndicator size="large" color="#ffd600"/><Text style={styles.wordStyle}>Fetching news...</Text></View>):(<View style={styles.flatList} >
        <FlatList 
        data = {topWords}
        renderItem = {({item}) =>(
          <TouchableOpacity style={styles.item} title = {item.word}
            key={item.word}
            onPress = {() => navigation.navigate('Second',{
              tweets_data : tweets[item.word],
              word : item.word,
              info : item.info
            })}
          >
          <Text style={styles.wordStyle}>{item.word}</Text>  
          </TouchableOpacity>
        )}
        keyExtractor ={(item, index)=> index.toString()}
      />
      </View>)}
      </View>

   </View>
  );
};
 
export default HomeScreen;