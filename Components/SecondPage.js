import React from 'react';
// import { useEffect } from 'react';
// import { FlatList } from 'react-native';
import { View, Text, Button, StyleSheet,ScrollView, Dimensions, Linking, Icon, Image  } from 'react-native';
import {Card } from 'react-native-elements'
import {ProgressBar} from 'react-native-multicolor-progress-bar';

const { width } = Dimensions.get('window');

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#D5F5E3',
    // alignItems: 'center',
    // justifyContent: 'center',
    
  },
  headlineView:{
    justifyContent : 'center',
    alignItems: 'center',
    marginBottom : 10,
    marginTop : 10
  },
  newsView: {
    marginTop: 0,
    backgroundColor: 'white',
    width: width - 80,
    margin: 10,
    height: 350,
    borderRadius: 10,
    //paddingHorizontal : 30
  }, 
  tweetsView: {
    marginTop: 0,
    backgroundColor: 'white',
    width: width - 80,
    margin: 10,
    height: 300,
    borderRadius: 10,
    //paddingHorizontal : 30
  },
  headline: {
    textAlign: 'center',
    fontWeight: 'bold',
    fontSize: 18,
    marginTop: 0,
    width: 250,
    backgroundColor: 'yellow',
  },
  profileImg: {
    height: 50,
    width: 50,
    borderRadius: 40,
    marginRight : 10
  },
  profileData:{ 
    justifyContent : 'center',
    alignItems: 'center',
    flexDirection: 'row',
    marginBottom:10
  }
});

const SecondScreen = ({route}) => {
//     const [titles, setTitles] = React.useState('');
//     const [links, setLinks] = React.useState('');
//     useEffect(()=>{
//     var title_array = []
//     var link_array = []
//       JSON.parse(route.params.values)['title'].forEach(element => {
//           title_array.push(element)
//       });
      
//       JSON.parse(route.params.values)['link'].forEach(element => {
//         link_array.push(element)
//     });
//     console.log("title array:", title_array)
//     console.log("link array:", link_array)
//     setTitles(title_array);
//     setLinks(link_array);
//   })
  return (
    
    <View style={styles.container}>
      <ScrollView style={styles.container}>
      <View style={styles.headlineView}>
      <Text style = {styles.headline}>{route.params.word.toString().toUpperCase()}</Text>
      </View>
      <ScrollView 
        // ref={(scrollView) => { this.scrollView = scrollView; }}
        style={styles.container}
        //pagingEnabled={true}
        horizontal= {true}
        decelerationRate={0}
        snapToInterval={width - 60}
        snapToAlignment={"center"}
        contentInset={{
          top: 0,
          left: 30,
          bottom: 0,
          right: 30,
        }}>
          {
               route.params.info.map((w, i) =>
                
               <View key={i} style={styles.newsView}>
                 <Card >
                    <Card.Title>{w.title}</Card.Title>
                    <Card.Divider/>
                    <Text numberOfLines={10} style={{ width: width - 150 }}> {w.summary} </Text>
                    <Button
                        // icon={<Icon name='code' color='#ffffff' />}
                        buttonStyle={{borderRadius: 0, marginLeft: 0, marginRight: 0, marginBottom: 0}}
                        title='VIEW NEWS'
                        onPress={() => Linking.openURL(`${w.link}`).catch((err)=>{
                          console.log(err);
                        })}/>
                </Card>
              
                 </View>              
               )
          }
      </ScrollView>
      {/* 'sentiment_score': {'neg': 0.248, 'neu': 0.648, 'pos': 0.104, 'compound': -0.9999}} */}
      {
        route.params.tweets_data['tweets'].length > 0 ? (
          <View>
          <View style={styles.headlineView}>

      <View style = {{ width: width - 80, paddingBottom: 20}}>
      {/* <Text> neg : {route.params.tweets_data['sentiment_score']['neg']} </Text>
      <Text> neu : {route.params.tweets_data['sentiment_score']['neu']} </Text>
      <Text> pos : {route.params.tweets_data['sentiment_score']['pos']} </Text> */}
      <View style= {{flexDirection: 'row', alignSelf: 'center', paddingBottom: 10}}>
          <Text style = {{fontWeight: 'bold', fontSize: 18, marginTop: 0, marginRight: 10}}> Public Tweets Attitude</Text>
          {
            route.params.tweets_data['sentiment_score']['compound'] > 0.667 ? (
                <Image source={require('../assets/thumbsup.png')} style={{height: 20, width: 20}} />
            ) : (
              <View></View>
            )
          }
          {
            route.params.tweets_data['sentiment_score']['compound'] < -0.667 ? (
              <Image source={require('../assets/thumbsdown.png')} style={{height: 20, width: 20}} />
            ) : (
              <View></View>
            )
          }
          {
            route.params.tweets_data['sentiment_score']['compound'] < 0.667 && route.params.tweets_data['sentiment_score']['compound'] > -0.667 ? (
              <Image source={require('../assets/neutral_emoji.png')} style={{height: 20, width: 20}} />
            ) : (
              <View></View>
            )
          }
      </View>
        <ProgressBar
          arrayOfProgressObjects={[
          {
            color: 'red',
            value: route.params.tweets_data['sentiment_score']['neg'],
            nameToDisplay: (route.params.tweets_data['sentiment_score']['neg']*100).toString().substr(0,5) + "%"
          },
          {
            color: '#D5DBDB',
            value: route.params.tweets_data['sentiment_score']['neu'],
            nameToDisplay: (route.params.tweets_data['sentiment_score']['neu']*100).toString().substr(0,5) + "%"
          },
          {
            color: '#00BFFF',
            value: route.params.tweets_data['sentiment_score']['pos'],
            nameToDisplay: (route.params.tweets_data['sentiment_score']['pos']*100).toString().substr(0,5) + "%"
          },
          ]}
        />
      </View>
      
      {/* <Text style = {styles.headline}>Tweets related to: {route.params.word}</Text> */}
      </View>
      <ScrollView 
        // ref={(scrollView) => { this.scrollView = scrollView; }}
        style={styles.container}
        //pagingEnabled={true}
        horizontal= {true}
        decelerationRate={0}
        snapToInterval={width - 60}
        snapToAlignment={"center"}
        contentInset={{
          top: 0,
          left: 30,
          bottom: 0,
          right: 30,
        }}>
          {
               route.params.tweets_data['tweets'].map((w, i) =>
               <View key={i} style={styles.tweetsView}>

                <Card >
                  <View style={styles.profileData}>
                  <Image source={{ uri:`${w.user['profile_image_url']}` }} style={styles.profileImg} />
                  <View>
                  <Card.Title>{w.user['name']}</Card.Title>
                  </View>
                   </View>
                    <Card.Divider/>
                    <Text numberOfLines={10} style={{ width: width - 150 }}> {w.text} </Text>
                    <Card.Divider/>
                    {/* <Text> Retweet Count : {w.retweet_count} </Text>
                    <Text> Likes : {w.like} </Text> */}
                    {/* <Text> {w.tweet_date} </Text> */}
                    {
                       `${w.url}` && (<Button
                      // icon={<Icon name='code' color='#ffffff' />}
                      buttonStyle={{borderRadius: 0, marginLeft: 0, marginRight: 0, marginBottom: 0}}
                      title='VIEW TWEET'
                      onPress={() => Linking.openURL(`${w.url}`).catch((err)=>{
                        alert("No Source URL")
                      })}/>) 
                    }
                </Card>
                 {/* <Text style={{color: 'blue'}}
                      onPress={() => Linking.openURL({w})}>
                            Google
                 </Text> */}
                 </View>
              //  <Card><Text key={i}> {w.title} </Text></Card>
                
               )
          }
      </ScrollView>
      </View>
        ) : (
          <Text style = {{textAlign: 'center', fontWeight: 'bold', fontSize: 18, marginTop: 0}}> No Tweets</Text>
        )
      }
      
      {/* <View>
      <View style={styles.headlineView}>

      <View style = {{ width: width - 80, paddingBottom:20}}>
      
      <Text style = {{textAlign: 'center', fontWeight: 'bold', fontSize: 18, marginTop: 0}}> Public Tweets Attitude</Text>
        <ProgressBar
          arrayOfProgressObjects={[
          {
            color: 'red',
            value: route.params.tweets_data['sentiment_score']['neg'],
            nameToDisplay: (route.params.tweets_data['sentiment_score']['neg']*100).toString().substr(0,5) + "%"
          },
          {
            color: '#D5DBDB',
            value: route.params.tweets_data['sentiment_score']['neu'],
            nameToDisplay: (route.params.tweets_data['sentiment_score']['neu']*100).toString().substr(0,5) + "%"
          },
          {
            color: 'blue',
            value: route.params.tweets_data['sentiment_score']['pos'],
            nameToDisplay: (route.params.tweets_data['sentiment_score']['pos']*100).toString().substr(0,5) + "%"
          },
          ]}
        />
      </View>
      
      <Text style = {styles.headline}>Tweets related to: {route.params.word}</Text>
      </View>
      <ScrollView 
        // ref={(scrollView) => { this.scrollView = scrollView; }}
        style={styles.container}
        //pagingEnabled={true}
        horizontal= {true}
        decelerationRate={0}
        snapToInterval={width - 60}
        snapToAlignment={"center"}
        contentInset={{
          top: 0,
          left: 30,
          bottom: 0,
          right: 30,
        }}>
          {
               route.params.tweets_data['tweets'].map((w, i) =>
               <View style={styles.tweetsView}>

                <Card >
                  <View style={styles.profileData}>
                  <Image source={{ uri:`${w.user['profile_image_url']}` }} style={styles.profileImg} />
                  <View>
                  <Card.Title>{w.user['name']}</Card.Title>
                  </View>
                   </View>
                    <Card.Divider/>
                    <Text numberOfLines={10} style={{ width: width - 150 }}> {w.text} </Text>
                    <Card.Divider/>
                    <Text> Retweet Count : {w.retweet_count} </Text>
                    <Text> Likes : {w.like} </Text>
                    
                    {
                       `${w.url}` && (<Button
                      // icon={<Icon name='code' color='#ffffff' />}
                      buttonStyle={{borderRadius: 0, marginLeft: 0, marginRight: 0, marginBottom: 0}}
                      title='VIEW TWEET'
                      onPress={() => Linking.openURL(`${w.url}`).catch((err)=>{
                        alert("No Source URL")
                      })}/>) 
                    }
                </Card>
                 
                 </View>
                
               )
          }
      </ScrollView>
      </View> */}
      
      </ScrollView>
    </View>
  );
};
 
export default SecondScreen;